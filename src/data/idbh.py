import math
import torch as tc
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode as Interpolation

class IDBH(tc.nn.Module):
    ColorBiased = [
        (0.125, 'color', 0.1, 1.9),
        (0.125, 'brightness', 0.5, 1.9),
        (0.125, 'contrast', 0.5, 1.9),
        (0.125, 'sharpness', 0.1, 1.9),
        (0.125, 'autocontrast'),
        (0.125, 'equalize'),
        (0.125, 'shear', 0.05, 0.15),
        (0.125, 'rotate', 1, 11)  
    ]
    ShapeBiased = [
        (0.08, 'color', 0.1, 1.9),
        (0.08, 'brightness', 0.5, 1.9),
        (0.04, 'contrast', 0.5, 1.9),
        (0.08, 'sharpness', 0.1, 1.9),
        (0.04, 'autocontrast'),
        (0.08, 'equalize'),
        (0.3, 'shear', 0.05, 0.35),
        (0.3, 'rotate', 1, 31)
    ]
    
    def __init__(self, space=None):
        super().__init__()

        self.space = {}
        p_accu = 0.0
        for trans in self.ColorBiased:
            p = trans[0]
            self.space[(p_accu, p_accu+p)] = trans[1:]
            p_accu += p
            
    def transform(self, img, trans):
        if len(trans) == 1:
            trans = trans[0]
        else:
            lower, upper = trans[1:]
            trans = trans[0]
            if trans == 'rotate':
                strength = tc.randint(lower, upper, (1,)).item()
            else:
                strength = tc.rand(1) * (upper-lower) + lower

        if trans == 'color':
            img = F.adjust_saturation(img, strength)
        elif trans == 'brightness':
            img = F.adjust_brightness(img, strength)
        elif trans == 'contrast':
            img = F.adjust_contrast(img, strength)
        elif trans == 'sharpness':
            img = F.adjust_sharpness(img, strength)
        elif trans == 'shear':
            if tc.randint(2, (1,)):
                # random sign
                strength *= -1
            strength = math.degrees(strength)
            strength = [strength, 0.0] if tc.randint(2, (1,)) else [0.0, strength]
            img = F.affine(img,
                           angle=0.0,
                           translate=[0, 0],
                           scale=1.0,
                           shear=strength,
                           interpolation=Interpolation.NEAREST,
                           fill=0)
        elif trans == 'rotate':
            if tc.randint(2, (1,)):
                strength *= -1
            img = F.rotate(img, angle=strength, interpolation=Interpolation.NEAREST, fill=0)
        elif trans == 'autocontrast':
            img = F.autocontrast(img)
        elif trans == 'equalize':
            img = F.equalize(img)

        return img
            
    def forward(self, img):
        roll = tc.rand(1)
        for (lower, upper), trans in self.space.items():
            if roll <= upper and roll >= lower:
                return self.transform(img, trans)
        
        return img
