import torch as tc

from torchattacks import *

from src.utils.printer import dprint
from src.utils.grad import input_grad

def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    adv = imgs.requires_grad_(True) if pert is None else tc.clamp(imgs+pert, 0, 1).requires_grad_(True)
    ig = input_grad(adv, targets, model, criterion) if ig is None else ig
    if pert is None:
        pert = eps_step*tc.sign(ig)
    else:
        pert += eps_step*tc.sign(ig)
    pert.clamp_(-eps, eps)
    adv = tc.clamp(imgs+pert, 0, 1)
    pert = adv-imgs
    return adv.detach(), pert.detach()

def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    for i in range(max_iter):
        adv, pert = perturb(imgs, targets, model, criterion, eps, eps_step, pert, ig)
        ig = None
    return adv, pert

ATTACK = {
    'FGM' : FGSM,
    'PGD' : PGD,
    'APGD' : APGD,
    'AA' : AutoAttack,
    'AA+' : AutoAttack
}

HP_MAP = {
    'n_classes' : 'out_dim',
    'steps' : 'max_iter',
    'alpha' : 'eps_step',
    'n_restarts': 'num_random_init',
    'loss' : 'adv_loss',
    'random_start' : 'num_random_init',
    'norm' : 'adv_norm',
    'version' : '-'
}

def fetch_attack(attack, model, **config):
    dprint('Adversary', **config)

    if attack == 'AA':
        config['version'] = 'standard'
    elif attack == 'AA+':
        config['version'] = 'plus'

    if 'seed' in config and config['seed'] is None:
        config['seed'] = 0

    return ATTACK[attack](model, **config)
    
