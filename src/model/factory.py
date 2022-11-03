import os, json
import torch as tc

from src.model.wide_resnet import Wide_ResNet
from src.model.preact_resnet import PreActResNet
from src.utils.printer import dprint

ARCHS = {'wresnet' : Wide_ResNet,
         'paresnet': PreActResNet}

def fetch_model(arch, checkpoint=None, **config):
    if arch in ARCHS:
        model = ARCHS[arch](**config)
    else:
        raise Exception("Invalid arch {}".format(arch))

    # extract checkpoint information
    if checkpoint is None:
        ck_lid = None
    elif isinstance(checkpoint, tuple):
        ck_lid, checkpoint = checkpoint
    elif os.path.isfile(checkpoint):
        ck_lid = checkpoint.split('/')[-1][:4]
    else:
        ck_lid = checkpoint

    hyper_params = model.hyperparams_log() if hasattr(model, 'hyperparams_log') else {}
    dprint('model',
           arch=arch,
           **hyper_params,
           checkpoint=ck_lid)
    
    if checkpoint is not None:
        # given checkpoint, resume the model from checkpoint
        if isinstance(checkpoint, str) and os.path.isfile(checkpoint):
            checkpoint = tc.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        
    return model


'''
Optimizer

'''

OPTIMS = {
    'sgd' : tc.optim.SGD,
    'adam' : tc.optim.Adam
}

def fetch_optimizer(optim, params, checkpoint=None, **args):
    # hyper-parameter report
    if checkpoint is not None:
        ck_lid, checkpoint = checkpoint
    else:
        ck_lid = None
    dprint('optimizer', optim=optim, checkpoint=ck_lid, **args)
    
    if optim in OPTIMS:
        optim = OPTIMS[optim](params, **args)
        if checkpoint is not None:
            optim.load_state_dict(checkpoint['optimizer'])
        return optim
    else:
        raise Exception("Invalid optimizer: {}".format(optim))
