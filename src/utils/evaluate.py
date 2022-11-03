import time
import torch as tc
from torch.utils.data import DataLoader
from torch.autograd import grad

from src.utils.helper import *
from src.utils.printer import sprint
from src.utils.adversary import pgd, perturb
from src.data.factory import fetch_dataset

def evaluate(args):
    fargs = args.func_arguments(fetch_model, postfix='arch')
    model = fetch_model(**fargs)

    fargs = args.func_arguments(fetch_criterion)
    criterion = fetch_criterion(**fargs)
    
    fargs = args.func_arguments(fetch_dataset)
    val_set = fetch_dataset(train=False, **fargs)
    loader = DataLoader(val_set,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=True)
    validate(loader, model, criterion, args)
    
def validate(loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':4.3f')
    losses = AverageMeter('Loss', ':.2e')
    acc1 = AverageMeter('Acc@1', ':6.2f')
    ign = AverageMeter('InGrad', ':.2e')
    fgsm = AverageMeter('FGSM', ':6.2f')
    pgdn = AverageMeter('PGD', ':6.2f')
    meters = [batch_time, losses, ign, acc1, fgsm, pgdn]
    progress = ProgressMeter(len(loader), meters, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    eps = args.eps
    eps_step = eps/4
    
    end = time.time()
    for i, (img, targets) in enumerate(loader, 1):
        img = img.to(args.device, non_blocking=True).requires_grad_(True)
        targets = targets.to(args.device, non_blocking=True)
        bs = img.size(0)
        
        output = model(img)
        loss = criterion(output, targets)
        acc1.update(accuracy(output, targets)[0][0], bs)
        losses.update(loss, bs)
        ig = grad(loss, img)[0]
        ign.update(tc.norm(ig, p=1), bs)
        
        if args.eval_iter > 0:
            adv, _ = perturb(img, targets, model, criterion, eps, eps, ig=ig)
            fgsm.update(accuracy(model(adv), targets)[0][0], bs)
            
            adv, _ = pgd(img, targets, model, criterion, eps, eps_step, args.eval_iter, ig=ig)
            pgdn.update(accuracy(model(adv), targets)[0][0], bs)
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i == 1 or i % args.log_pbtc == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    if args.world_size > 1:
        data = [acc1, ign, fgsm, pgdn]
        data_sum = reduce_all(args.rank, tc.tensor([d.sum for d in data]).to(args.device))
        data_count = reduce_all(args.rank, tc.tensor(acc1.count).to(args.device))
        if data_sum is not None:
            for d, d_sum in zip(data, data_sum):
                d.avg = d_sum / data_count

    sprint(' *  Acc@1: {0.avg:.2f} | FGSM: {1.avg:.2f} | PGD: {2.avg:.2f} | IG: {3.avg:.3e}'
           .format(acc1, fgsm, pgdn, ign))
    return acc1.avg.item(), ign.avg.item(), fgsm.avg.item(), pgdn.avg.item()
