import os, sys
import numpy as np
from argparse import ArgumentParser

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config.config import Configuration, PATH, DATA, upper, pixel_2_real
from src.config.config import PARSER as SHARED
from src.utils.log import ids_from_idx, idx_from_ids, complete_ids

parser = ArgumentParser(parents=[SHARED])
parser.add_argument('--train', action='store_true', default=False,
                    help="evaluate on training set")
parser.add_argument('--augment', nargs='+', default=None,
                   help="input data augmentation")
parser.add_argument('--mode', choices=['train', 'eval'], default='eval',
                    help="train or eval mode for models")

parser.add_argument('log_ids', nargs='+',
                    help="models to be tested based on the log ids")
parser.add_argument('-v', '--version', choices=['acc', 'swa_pgd', 'swa_end', 'pgd', 'end'], default='pgd',
                    help="the version of model for the specified log id to be evaluated")

parser.add_argument('-t', '--transfer',
                    help="the log id of model whose adversarial examples to be transferred")

parser.add_argument('-a', '--attack', choices=['FGSM', 'AA','PGD', 'CW', 'APGD'],
                    type=upper,
                    default=None,
                    help="attack method for crafting adversarial examples")
parser.add_argument('--adv_loss', choices=['ce', 'dlr'], default='ce',
                    help="loss used to compute input gradients")
parser.add_argument('--eps', type=pixel_2_real, default=pixel_2_real(8),
                    help="attack strength for crafting adversarial examples")
parser.add_argument('--max_iter', type=int, default=50,
                    help="the maximum number of iterations for optimizing adversarial examples")
parser.add_argument('--adv_norm', choices=['Linf', 'L2'], default='Linf',
                    help="norm used to constrain the adversary")
parser.add_argument('--eps_step', type=pixel_2_real, default=None,
                    help="step size for multi-step attacks")
parser.add_argument('--nrandom_init', dest='num_random_init', type=int, default=0,
                    help="num of random iniailization when generating adversarial examples")

def parse_ids(ids):
    log_ids = []
    for log_id in ids:
        if '-' in log_id:
            start_ids, end_ids = log_id.split('-')
            start_idx = idx_from_ids(start_ids)
            end_idx = idx_from_ids(end_ids)
            log_ids += [ids_from_idx(idx) for idx in range(start_idx, end_idx+1)]
        else:
            log_ids.append(complete_ids(log_id))
    return log_ids

class AdversaryConfig(Configuration):
    def __init__(self):
        super(AdversaryConfig, self).__init__(parser)

        # parse log ids
        self.log_ids = parse_ids(self.log_ids)
        
        self.nb_random_init = self.num_random_init
