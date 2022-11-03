import torch as tc
import os, glob, sys
import numpy as np
from uuid import uuid4
from argparse import ArgumentParser

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config.config import Configuration, DATA, PATH, upper, pixel_2_real
from src.config.config import PARSER as SHARED

parser = ArgumentParser(parents=[SHARED])

# Model
archs = ["wresnet", "paresnet"]

parser.add_argument('-a', '--arch', choices=archs, default='wresnet',
                    help="the neural network architecture to be used")
parser.add_argument('--width', type=int, default=1,
                    help="the width factor for widening conv layers")
parser.add_argument('--checkpoint', default=None,
                    help="the checkpoint of model to be loaded from")
parser.add_argument('--depth', type=int, default=34,
                    help="depth of wide resnet")
parser.add_argument('--activation', choices=['relu', 'softplus'], default='relu',
                    help="nonlinear activation function")


# Dataset
parser.add_argument('-d', '--dataset', choices=['TIN', 'CIFAR10', 'SVHN'], type=upper, default='CIFAR10',
                    help="the dataset to be used")
parser.add_argument('--download', action='store_true',
                    help="download dataset if not exists")
parser.add_argument('--idbh', choices=['cb', 'sb'], default='cb',
                   help="")
parser.add_argument('--cropshift', nargs='+', type=int, default=(0, 10),
                   help="lower and upper bound to sample the strength of cropshift")


# Learning Arguments
parser.add_argument('--lr', type=float, default=0.1,
                   help="learning rate in optimizer")
parser.add_argument('--annealing', nargs='+', default=[100, 150],
                   help="learning rate decay every N epochs")
parser.add_argument('--momentum', type=float, default=0.9,
                   help="momentum in optimizer")
parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4,
                   help="weight decay in optimizer")
parser.add_argument('-e', '--epochs', type=int, default=200,
                   help="maximum amount of epochs to be run")
parser.add_argument('--augment', nargs='+', default=[],
                   help="input data augmentation")
parser.add_argument('--swa', nargs='+', default=None,
                   help="enable stochastic weight averaging")

parser.add_argument('--optim', choices=['sgd', 'adam'], default='sgd',
                   help="optimizer")
parser.add_argument('--nesterov', action='store_true',
                   help="enable nesterov momentum for the optimizer")

'''
Adversray Training

'''

parser.add_argument('-na', '--non-adversary', action='store_false', dest='advt', default=True,
                    help="disable adversarial training")
parser.add_argument('--eps', type=pixel_2_real, default=None,
                    help="attack strength i.e. constraint on the maxmium distortion")
parser.add_argument('--max_iter', type=int, default=1,
                    help="maximum iterations for generating adversary")
parser.add_argument('--eps_step', type=pixel_2_real, default=None,
                    help="step size for multi-step attacks")
parser.add_argument('-ei', '--eval_iter', type=int, default=10,
                    help="number of steps to generate adversary in the main procedure")
parser.add_argument('-ws', '--warm_start', action='store_true', default=False,
                    help="gradually increase perturbation budget in the first 5 epochs")


class TrainConfig(Configuration):
    def __init__(self):
        super(TrainConfig, self).__init__(parser)
            
        if self.resume is None:
            # temporary id
            self.tmp_id = str(uuid4())
        else:
            self.logger.log_id = self.resume
            log = self.logger[self.resume]
            # resume configuration from log
            log['model']['checkpoint'] = self.resume
            configs = {**log['training'], **log['model'], **log['dataset']}
            for k, v in configs.items():
                if hasattr(self, k):
                    default = parser.get_default(k)
                    if getattr(self, k) == default:
                        setattr(self, k, v)
                        
        # all hyper-parameters to be recorded should be specified above
        if self.logging and self.resume is None:
            self.logger.new(self)

        if self.checkpoint is not None:
            self.checkpoint = self.path('trained', "{}/{}_end".format(self.logbook, self.checkpoint))

        if self.parallel:
            self.batch_size = int(self.batch_size / self.world_size)
            
        if self.swa is not None:
            self.swa_start = int(self.swa[0])
            self.swa_decay = float(self.swa[1]) if self.swa[1] != 'n' else self.swa[1]
            self.swa_freq = int(self.swa[2])
            
    @property
    def log_id(self):
        lid = self.logger.log_id
        return self.tmp_id if self.logger.log_id is None else lid

    @property
    def log_required(self):
        return self.logging or self.resume
