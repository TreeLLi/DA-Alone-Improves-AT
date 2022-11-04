import os, sys, time
import numpy as np
from addict import Dict
from itertools import product

from torchattacks import *

import torch as tc
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.autograd import grad
from statistics import mean

from config.config import real_2_pixel
from config.adversary import *
from src.model.factory import fetch_model, ARCHS
from src.data.factory import fetch_dataset, DATASETS
from src.utils.helper import accuracy, gather_all, run, AverageMeter, ProgressMeter
from src.utils.printer import sprint
from src.utils.adversary import *

def attack(args):
    for lid in args.log_ids:
        try:
            _attack(lid, args)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            sprint("Adversary experiment {} failed.\nError: {}.".format(lid, sys.exc_info()[1]))
            raise
            
def _attack(lid, args):
    logger = args.logger
    transfer = args.transfer is not None
    log = logger.fetch(lid)
    args.dataset = log.dataset.dataset

    fargs = args.func_arguments(fetch_dataset, DATASETS, postfix='data')
    dataset = fetch_dataset(**fargs)
    if args.world_size > 1:
        split = [len(d) for d in dataset]
        acc_nsamples = [sum(split[:i+1]) for i in enumerate(split)]
        dataset = dataset[args.rank]
    loader = DataLoader(dataset,
                        batch_size=args.batch_size)

    model_lid = args.transfer if transfer else lid
    version = 'acc' if transfer else args.version
    model_info = logger.fetch(model_lid).model
    model_path = args.path('trained', '{}/{}_{}'.format(args.logbook, model_lid, version))
    arch = model_info.arch if model_info.arch in ARCHS else 'custom'
    fargs = args.func_arguments(fetch_model, ARCHS[arch], postfix='arch')
    model_info['checkpoint'] = model_path
    model = fetch_model(**model_info, **fargs)
    model = model.cuda()
    if args.mode == 'train':
        model.train()
    else:
        model.eval()
    
    if transfer:
        attack_base = (model_lid, model)
        model_path = args.path('trained', '{}/{}_{}'.format(args.logbook, lid, args.version))
        model_info['checkpoint'] = model_path
        model = fetch_model(**model_info, **fargs)
        model = model.cuda()
        model.eval()
    else:
        attack_base = model

    if args.attack is not None:
        fargs = args.func_arguments(fetch_attack, ATTACK, trans=HP_MAP)
        attack = fetch_attack(model=model, **fargs)

    criterion = tc.nn.CrossEntropyLoss()

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    acc = AverageMeter('Acc@1', ':6.2f')
    meters = [batch_time, losses, acc]

    progress = ProgressMeter(len(loader), meters, prefix="Batch: ")
    
    sprint("Adversary evaluation starts!", True)
    
    end = time.time()
    for i, data in enumerate(loader, 1):
        imgs = data[0].to(args.device, non_blocking=True)
        labels = data[1].to(args.device, non_blocking=True)
        batch_size = imgs.size()[0]

        if args.attack is not None:
            imgs = attack(imgs, labels).detach()

        output = model(imgs)
        loss = criterion(output, labels)
            
        acc.update(accuracy(output, labels, topk=(1,))[0], batch_size)
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i == 1 or i % args.log_pbtc == 0:
            progress.display(i)
            
    if args.world_size > 1:
        gathered = gather_all(args.rank, args.world_size, acc1)
        print("gather finished on {}".format(args.rank))
        if gathered is not None:
            print(gathered)
            for _acc1 in gathered[1:]:
                update(acc1, *_acc1)

    sprint("* Acc@1: {:.2f}".format(acc.avg.item()))
    if args.logging:
        attack = args.attack

        if attack is None:
            info = {args.version : '{:3.2f} L: {:3.2f}'.format(acc.avg.item(), losses.avg)}
            logger.update('checkpoint', **info, ids=lid, save=True)
        else:
            if transfer:
                attack = '{}_{}'.format(attack, args.transfer)
            strength = "{}_{}_{}".format(args.version, real_2_pixel(args.eps), args.num_random_init)
            if args.attack == 'PGD' or args.attack == 'APGD':
                strength += '_{}'.format(args.max_iter)
            logger.update('robustness', attack, **{strength:acc.avg.item()}, ids=lid, save=True)
            
if __name__ == '__main__':
    cfg = AdversaryConfig()
    
    run(attack, cfg)
