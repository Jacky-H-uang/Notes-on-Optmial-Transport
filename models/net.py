import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from .ot_clip import CustomCLIP, get_cfg_defaults

class make_model(nn.Module) :
    def __init__(self,args):
        super(make_model, self).__init__()

        self.cfg = get_cfg_defaults()

        self.model = CustomCLIP(classnames=args.classnames, clip_model=args.clip_model)

    def forward(self, img):

        return self.model(img)
