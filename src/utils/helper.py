import random
import shutil
import torch as tc
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from warnings import warn

import src.utils.printer as printer

def run(fn, args):
    if args.seed is not None:
        random.seed(args.seed)
        tc.manual_seed(args.seed)
        cudnn.deterministic = True
        warn('You have chosen to seed training. '
             'This will turn on the CUDNN deterministic setting, '
             'which can slow down your training considerably! '
             'You may see unexpected behavior when restarting '
             'from checkpoints.')

    cudnn.benchmark = True
    if args.parallel:
        tc.multiprocessing.spawn(init_dist, nprocs=args.nprocs, args=(fn, args))
    else:
        args.track_signals()
        fn(args)
        args.end()
        
def init_dist(rank, fn, args):
    args.track_signals()
    
    args.use_device(rank)
    args.rank = rank
    printer.set_rank(rank)
    print("worker {} runs on the device {}".format(rank, args.device))
    
    fargs = args.func_arguments(dist.init_process_group)
    dist.init_process_group(**fargs)
    
    fn(args)

    args.end()

# gather target from all processes to rank-0 process
# asynchroneous for GPU target since GPU ops are all asynchroneous
# not support for NCCL distribution backend
def gather_all(rank, world_size, target):
    if rank == 0:
        gathered = [target.clone() for _ in range(world_size)]
        dist.gather(target, gathered)
        return gathered
    else:
        dist.gather(target)

# sum target from all processes to rank-0 process
def reduce_all(rank, target):
    dist.reduce(target, 0)
    if rank == 0:        
        return target
    
def save_checkpoint(state, is_best_acc, is_best_robust, args):
    lid = args.logger.log_id
    eid = args.experiment_id
    id = lid if lid is not None and lid != eid else eid
    
    ck_filepath = args.path('checkpoint', id)
    tc.save(state, ck_filepath)
    root = 'trained' if args.logging else 'tmp'
    file_name = "{}/{}".format(args.logbook, id) if args.logging else id
    if is_best_acc:
        shutil.copyfile(ck_filepath, args.path(root, file_name+'_acc'))
    if is_best_robust:
        shutil.copyfile(ck_filepath, args.path(root, file_name+'_rob'))
    
'''
Logger

'''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        kvs = {k:v.item() if isinstance(v, tc.Tensor) else v for k, v in self.__dict__.items()}
        return fmtstr.format(**kvs)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

'''
Calculator

'''

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
