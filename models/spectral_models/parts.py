"""
 Created by Narayan Schuetz at 07/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


from spectral import Dft2d, iDft2d, DctII2d, iDctII2d, DctII1d
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


# ----------------------------------------------------------------------------------------------------------------------
# Model parts are defined here
# ----------------------------------------------------------------------------------------------------------------------


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class DiscreteCosine2dConvBlockHybrid(nn.Module):
    """
    Defines a Discrete Cosine 2D Hybrid Block. It performs Conv -> Cos2D -> DepthConcat -> 1x1 operations on the input.
    """

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 kernel_size_pooling=None,
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.cos = DctII2d(width, height, fixed=fixed)
        self._1x1 = nn.Conv2d(in_channels=out_channels+in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        conv_out = self.conv(x)
        spec_out = self.cos(x)
        out = torch.cat((conv_out, spec_out), 1)
        out = self._1x1(out)
        return out


class DiscreteCosine2dConvBlock(nn.Module):

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 kernel_size_pooling=None,
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.cos = DctII2d(width, height, fixed=fixed)

    def forward(self, x):
        x = self.conv(x)
        x = self.cos(x)
        return x


class DiscreteFourier2dConvBlockHybrid(nn.Module):
    """
    Defines a Discrete Fourier 2D Hybrid Block. It performs Conv -> Dft2D -> DepthConcat -> 1x1 operations on the input.
    """

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 kernel_size_pooling=None,
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.fft = Dft2d(width, height, fixed=fixed)
        self._1x1 = nn.Conv2d(in_channels=out_channels+in_channels*2, out_channels=out_channels, kernel_size=1, groups=1)

    def forward(self, x):
        conv_out = self.conv(x)
        spec_out = self.fft(x)
        out = torch.cat((conv_out, spec_out), 1)
        out = self._1x1(out)
        return out


class DiscreteFourier2dConvBlock(nn.Module):

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 kernel_size_pooling=None,
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.dft = Dft2d(width, height, fixed=fixed)

    def forward(self, x):
        x = self.conv(x)
        x = self.dft(x)
        return x


class InverseDiscreteCosine2dConvBlockHybrid(nn.Module):
    """
    Defines an inverse Discrete Cosine 2D Hybrid Block. It performs Conv -> iCos2D -> DepthConcat -> 1x1 operations on
    the input.
    """

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 kernel_size_pooling=None,
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.cos = iDctII2d(width, height, fixed=fixed)
        self._1x1 = nn.Conv2d(in_channels=out_channels+in_channels, out_channels=out_channels, kernel_size=1, groups=1)

    def forward(self, x):
        conv_out = self.conv(x)
        spec_out = self.cos(x)
        out = torch.cat((conv_out, spec_out), 1)
        out = self._1x1(out)
        return out


class InverseDiscreteCosine2dConvBlock(nn.Module):

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 kernel_size_pooling=None,
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.cos = iDctII2d(width, height, fixed=fixed)

    def forward(self, x):
        x = self.conv(x)
        x = self.cos(x)
        return x


class InverseDiscreteFourier2dConvBlockHybrid(nn.Module):
    """
    Defines an inverse Discrete Fourier 2D Hybrid Block. It performs Conv -> iDct2D -> DepthConcat -> 1x1 operations on
    the input.
    """

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 kernel_size_pooling=None,
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.ifft = iDft2d(width, height, fixed=fixed)
        self._1x1 = nn.Conv2d(in_channels=out_channels+in_channels, out_channels=out_channels, kernel_size=1, groups=1)

    def forward(self, x):
        conv_out = self.conv(x)
        spec_out = self.ifft(x)
        out = torch.cat((conv_out, spec_out), 1)
        out = self._1x1(out)
        return out


class InverseDiscreteFourier2dConvBlock(nn.Module):

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 kernel_size_pooling=None,
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.dft = iDft2d(width, height, fixed=fixed)

    def forward(self, x):
        x = self.conv(x)
        x = self.dft(x)
        return x


class ConvBlock(nn.Module):
    """
    Defines a block doing just regular convolutions, to keep parameters approximately similar to the other blocks,
    the number of filters of the normal conv were doubled.
    """

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 kernel_size_pooling=None,
                 padding=1,
                 groups_conv=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        return out



