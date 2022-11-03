import os, sys, signal
from inspect import getargspec, signature
from pickle import dump
import torch as tc
from warnings import warn
from addict import Dict
from pynvml import *
from argparse import ArgumentParser

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.utils.log import Logger

'''
PATH 

'''        

PATH = Dict()

PATH.ROOT = root_path
PATH.DATA = os.path.join(root_path, 'data')
PATH.OUTPUT = os.path.join(root_path, 'output')
PATH.MODEL = os.path.join(root_path, 'model')
PATH.TMP = os.path.join(root_path, 'tmp')

PATH.LOG = os.path.join(PATH.OUTPUT, 'log')
PATH.CHECKPOINT = os.path.join(PATH.TMP, 'checkpoint')
PATH.TRAINED = os.path.join(PATH.MODEL, 'trained')
PATH.ARCHITECTURE = os.path.join(PATH.MODEL, 'architecture')
PATH.ANALYSIS = os.path.join(PATH.MODEL, 'analysis')
PATH.FIGURE = os.path.join(PATH.OUTPUT, 'figure')


'''
DATASET

'''

DATA = Dict()

DATA.MNIST.stat = ((0.1307,), (0.3081,))
DATA.MNIST.dim = (1, 28, 28)
DATA.MNIST.nclasses = 10
DATA.MNIST.eps = 0.3
DATA.MNIST.eps_step = 0.01

DATA.CIFAR10.stat = ((0.4914008984375, 0.482159140625, 0.446531015625),
                     (0.24703278185799551, 0.24348423011049403, 0.26158752307127053))
DATA.CIFAR10.dim = (3, 32, 32)
DATA.CIFAR10.nclasses = 10
DATA.CIFAR10.eps = 8/255
DATA.CIFAR10.eps_step = 2/255

DATA.SVHN.dim = (3, 32, 32)
DATA.SVHN.nclasses = 10
DATA.SVHN.eps = 8/255
DATA.SVHN.eps_step = 2/255

DATA.TIN.stat = ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
DATA.TIN.dim = (3, 64, 64)
DATA.TIN.nclasses = 200
DATA.TIN.eps = 8/255
DATA.TIN.eps_step = 2/255


'''
Hyper-parameters

'''

PARSER = ArgumentParser(add_help=False,
                        description='Hyper-parameters shared among different experiments')

PARSER.add_argument('-bs', '--batch_size', type=int, default=128,
                   help="number of images in each mini-batch")
PARSER.add_argument('--num_workers', type=int, default=0,
                    help="number of processes used by each DataLoader to load data")
PARSER.add_argument('--seed', type=int,
                    help="setting random seed to fix the random generator")
PARSER.add_argument('--job_id',
                    help="the job id from Slurm")
PARSER.add_argument('--resume', default=None,
                    help="the checkpoint to be resumed from")

# Training Devices and Parallel
# defaul parallel paradigm is Distributed Parallel a.k.a. multiprocessing parallel
PARSER.add_argument('--cpu', dest='device', action='store_false',
                    help="force the program to use CPU while the default device is GPU")
PARSER.add_argument('--parallel', action='store_true',
                    help="enable multiprocessing parallel computing")
PARSER.add_argument('--nprocs', type=int, default=0,
                    help="number of subprocesses to be used in parallel computing")
PARSER.add_argument('--world_size', type=int, default=0,
                    help="total number of devices to be distributed on")
PARSER.add_argument('-r', '--local_rank', dest='rank', type=int, default=None,
                    help="the specific device to be used")
PARSER.add_argument('--init_method',
                    help='url used when distributed among various machines')

# Debugging
PARSER.add_argument('--log_pbtc', type=int, default=25,
                    help="log training info every N batches")
PARSER.add_argument('--nlogging', dest='logging', action='store_false',
                    help="stop logging experiment detail and result")
PARSER.add_argument('-l', '--logbook', default='log',
                    help="the name of log book to be used.")


'''
conversion

'''

def pixel_2_real(pixel):
    return float(pixel) / 255

def real_2_pixel(real):
    return int(real*255)

def upper(x):
    if isinstance(x, str):
        return x.upper()
    else:
        return str(x).upper()


'''
Configuration Manager 

'''

class Configuration:
    def __init__(self, parser):
        parser.parse_args(namespace=self)
        
        self.arch_root = PATH.ARCHITECTURE
        self.data_root = PATH.DATA

        if self.log_required:
            self.logger = Logger(self.path('log', '{}.json'.format(self.logbook)))
        
        # setting appropriate devices and parallel mode based on the available resources
        # self.device holds bool value if using gpu in argparse
        if self.device:
            self.device = 'cuda'
            self.avail_cudas = get_available_cudas()
            ncudas = min(tc.cuda.device_count(), len(self.avail_cudas))
            if self.avail_cudas == []:
                self.device = 'cpu'
                print("!-> CUDA is not available so swich to CPU device.")
            elif ncudas>1 and not self.parallel:
                # using the first cuda with maximum amount of free memory
                free_mem = get_memory_info(self.avail_cudas)
                max_free_idx = free_mem.index(max(free_mem))
                self.use_device(max_free_idx)
        else:
            self.device = 'cpu'
        
        # setting parallel parameters
        if self.parallel:
            # multiprocessing distributed compute
            # currently only concern one machine multiple devices case
            # ignoring network distributed case
            self.ncpus = tc.multiprocessing.cpu_count()
            max_nprocs = self.ncpus if self.using_cpu() else min(self.ncpus, ncudas)
            self.nprocs = min(self.nprocs, max_nprocs) if self.nprocs>0 else max_nprocs
            if self.nprocs <= 1:
                self.parallel = False
                warn("Data parallel is disabled due to no multiple devices available")
            else:
                self.world_size = self.nprocs
                # nccl for distributed GPU training and gloo for CPU case following PyTorch official guide.
                self.backend = 'gloo' if self.using_cpu() else 'nccl'
        else:
            self.rank = 0
            
        # report experiment configuration
        if self.parallel:
            print("=> Multiprocessing distributed on {} {}s".format(self.world_size, self.device))
        else:
            print("=> Running on {}".format(self.device))

    def using_cpu(self):
        return self.device == 'cpu'
        
    def use_device(self, idx):
        if self.using_cpu() or idx is None:
            return

        # rank idx is consecutive integer ranging from 0 to world_size
        # cuda idx is the idx of available cuda so not necessarily equivalent
        cuda_idx = str(self.avail_cudas[idx])
        self.device = 'cuda:' + cuda_idx
        tc.cuda.set_device(self.device)
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_idx

    @property
    def dataset(self):
        return self.__dataset
    
    @dataset.setter
    def dataset(self, dataset):
        self.__dataset = dataset
        dataset_info = DATA[self.dataset]
        self.data_stat = dataset_info.stat
        self.input_dim = dataset_info.dim
        self.out_dim = dataset_info.nclasses

        if hasattr(self, 'eps') and (self.eps is None or self.eps == []):
            self.eps = dataset_info.eps

        if hasattr(self, 'eps_step') and self.eps_step is None:
            self.eps_step = dataset_info.eps_step
            
        if hasattr(self, 'eps_step1') and self.eps_step1 is None:
            self.eps_step1 = dataset_info.eps

    @property
    def log_required(self):
        return True
            
    def func_arguments(self, fn, nested=None, trans=None, prefix='', postfix='', **params):
        return func_arguments_from(fn, self, nested, trans, prefix, postfix, **params)

    def abstract(self):
        return ''
        
    def path(self, root, name=None, ext='.pth.tar'):
        if name is not None and '.' not in name and ext is not None:
            name += ext
        return get_path(root, name)

    def dir(self, root, subpath=None):
        if subpath is not None and subpath[-1] != '/':
            subpath += '/'
        return get_path(root, subpath)
    
    def track_signals(self):
        signal.signal(signal.SIGINT, self.handler)
        signal.signal(signal.SIGUSR1, self.handler)
        signal.signal(signal.SIGUSR2, self.handler)

    # handle unexpected exits: KeyInterrupt etc.
    def handler(self, signum, frame):
        self.end(signum)
        
        if self.rank is not None:
            sys.exit("process {} exits.".format(self.rank))
        else:
            sys.exit()
        
    def end(self, reason=None):
        if self.rank is not None and self.rank != 0:
            # only end once on process 0 while multiprocessing
            return

        if reason is not None:
            if reason == signal.SIGUSR1:
                print("Program aborted due to the NaN loss.")
            elif reason == signal.SIGUSR2:
                print("Early stop training due to gradient masking.")
                
        if self.logging:
            self.logger.save()
            if self.logger.log_id is not None:
                print("Result saved in the log: {}".format(self.logger.log_id))
    
def func_arguments_from(fn, source, nested=None, trans=None, prefix='', postfix='', **params):
    fn_args = str(signature(fn)).replace(' ', '')[1:-1]
    fn_args = fn_args.split(',')

    print(fn_args)
    
    if 'self' in fn_args:
        # class initializer
        fn_args.remove('self')
        
    args = Dict()
    for arg in fn_args:
        if '=' in arg: arg = arg[:arg.index('=')]
        if ':' in arg: arg = arg[:arg.index(':')]
        if '*' in arg: continue
        
        if arg.startswith(prefix):
            arg = arg[len(prefix):]
            attr = trans[arg] if trans and arg in trans else arg
            pattr = '{}_{}'.format(postfix, attr)
        if hasattr(source, attr):
            args[arg] = getattr(source, attr)
        elif hasattr(source, pattr):
            args[arg] = getattr(source, pattr)
        elif hasattr(source, '__getitem__'):
            if attr in source:
                args[arg] = source[attr]
            elif pattr in source:
                args[arg] = source[pattr]
    if nested:
        if isinstance(nested, dict):
            cls_id = args[fn_args[0]]
            fn = nested[cls_id] if cls_id in nested else nested['custom']
        else:
            fn = nested
        args.update(func_arguments_from(fn, source, trans=trans, prefix=prefix, postfix=postfix))
        
    if params is not None:
        args.update(params)
    
    return args

def get_path(root, filename=None):
    if root.upper() in PATH:
        root = PATH[root.upper()]

    if filename is not None:
        root, filename = os.path.join(root, filename).rsplit('/', 1)    
        
    if not os.path.isdir(root):
        os.makedirs(root)
        print("Directory, {}, not exists and is created now.".format(root))

    if filename:
        return os.path.join(root, filename)
    else:
        return root


'''
CUDA

'''

# number of cudas not occupied by other processes
def get_available_cudas():
    if tc.cuda.is_available():        
        nvmlInit()
        ncudas = nvmlDeviceGetCount()
        acudas = []
        for i in range(ncudas):
            cuda = nvmlDeviceGetHandleByIndex(i)
            procs = nvmlDeviceGetComputeRunningProcesses(cuda)
            # if procs == []:
            acudas.append(i)
        nvmlShutdown()
        return acudas
    else:
        return []

def get_memory_info(cuda_idx):
    free_rates = []
    nvmlInit()
    for cid in cuda_idx:
        cuda = nvmlDeviceGetHandleByIndex(cid)
        mem_info = nvmlDeviceGetMemoryInfo(cuda)
        free_rates.append(mem_info.free / mem_info.total)
    nvmlShutdown()
    return free_rates
