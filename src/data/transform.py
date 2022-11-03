import math
import torch as tc
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode as Interpolation

class Transform(tc.nn.Module):
    ONLY = None
    INTERPOLATION = {
        'nearest' : 
        'bilinear' : Interpolation.BILINEAR
    }
    def __init__(self, strength=None, signed=False, interpolation='nearest', fill=0):
        super().__init__()
        self.signed = signed
        assert interpolation in self.INTERPOLATION
        self.interpolation = self.INTERPOLATION[interpolation]
        self._fill = fill if fill in ['img', 'uniform'] else int(fill)
        self.strength = self.parse_arg(strength)
        
    def __str__(self):
        return "{}/strength:{}/signed:{}".format(type(self).__name__, self.strength, self.signed)

    def parse_arg(self, arg):
        if arg is None:
            return 0
        
        if isinstance(arg, str):
            arg = arg.split('-')
            assert len(arg) <= 2
            if len(arg) == 1:
                return float(arg[0])
            else:
                return (float(arg[0]), float(arg[1]))

    def is_tensor(self, img):
        return isinstance(img, tc.Tensor)
            
    def sample_arg(self, arg, signed=False, integer=False):
        if isinstance(arg, tuple):
            low, high = arg
            if integer:
                low, high = int(low), int(high)
                arg = tc.randint(low, high, (1,)).item()
            else:
                arg = (tc.rand(1) * (high-low) + low).item()
            
        if signed and tc.randint(2, (1,)):
            arg *= -1

        return int(arg) if integer else arg
        
    def get_strength(self, integer=False, signed=None):
        signed = self.signed if signed is None else signed
        return self.sample_arg(self.strength, signed, integer)
        
    def fill(self, img, size=None):
        if self._fill == 'img':
            assert None not in [size, img]
            perm = tc.randperm(tc.numel(img))
            num = math.prod(size)
            idx = perm[:num]
            fill = img.view(-1)[idx].view(size).clone()
        elif self._fill == 'uniform':
            size = (1,) if size is None else size
            if isinstance(img, tc.Tensor):
                fill = tc.rand(size)
            else:
                fill = tc.randint(0, 256, size)
        else:
            fill = 1 if size is None else tc.ones(size)
            if isinstance(img, tc.Tensor):
                fill *= self._fill / 255.0
            else:
                fill *= self._fill

        if isinstance(fill, tc.Tensor) and fill.size() == (1,):
            return fill.item()
        else:
            return fill

