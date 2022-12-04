""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import scipy
from torch import nn
import functools
# from Networks.spectralnorm import SpectralNorm

class PatchGAN(nn.Module):
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,crop_center = None,FC_bottleneck=False):
        """Construct a PatchGAN discriminator  - #https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
        Modified to have double-convs, cropping, and a bottle-neck to use a vanilla dicriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            crop_center      -- None ot the size of the center patch to be cropped
            FC_bottleneck      -- If True use global average pooling and output a one-dimension prediction
        """
        self.crop_center = crop_center
        super(PatchGAN, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(ndf, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        if FC_bottleneck:
            sequence +=[nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(),nn.Linear(ndf * nf_mult,128),
                        nn.LeakyReLU(0.2, True),
                        nn.Linear(128,1)]
        else:
            sequence += [
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence).apply(self.weights_init)

    def forward(self, input):
        """Standard forward."""
        if self.crop_center is not None:
            _,_,h,w = input.shape
            x0 = (h-self.crop_center) //2
            y0 = (w-self.crop_center) //2
            input = input[:,:,x0:x0+self.crop_center,y0:y0+self.crop_center]
        return self.model(input)