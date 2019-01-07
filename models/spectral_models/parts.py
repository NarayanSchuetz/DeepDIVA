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


class Dct2dPooling(nn.Module):
    """
    Implements pooling of 2d DCT outputs. It works by only cropping the matrix around the top left corner and then
    transforms the non-cropped spectral parts back to the spatial domain. 
    The result is a "compressed" version of the original image.
    """

    def __init__(self, new_width, new_height):
        super().__init__()

        self.width = new_width
        self.height = new_height

    def forward(self, x):
        return x[:, :, :self.height, :self.width].clone()


class DctII2dComparison(nn.Module):
    """
    Discrete Cosine II 2D layer with randomly (a variant of Xavier uniform) initialized weights.
    """

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.weights_1 = nn.Parameter(self._create_weight_tensor())
        self.weights_2 = nn.Parameter(self._create_weight_tensor())

    def _create_weight_tensor(self, ):
        w = torch.Tensor(self.width, self.height)
        stdv = 1. / math.sqrt(w.size(1))
        w.uniform_(-stdv, stdv)
        return w

    def forward(self, input):
        x = F.linear(torch.transpose(input, -2, -1), self.weights_1)
        return F.linear(torch.transpose(x, -2, -1), self.weights_2)


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
                 groups_conv=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.cos = DctII2d(width, height, fixed=fixed)
        self._1x1 = nn.Conv2d(2 * out_channels, out_channels, 1, groups=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.cos(x)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
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
                 groups_conv=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.cos = DctII2d(width, height, fixed=fixed)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.cos(x)
        out = self.pool(out)
        return out


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
                 groups_conv=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.fft = Dft2d(width, height, fixed=fixed)
        self._1x1 = nn.Conv2d(3 * out_channels, out_channels, 1, groups=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.fft(x)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
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
                 groups_conv=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels//2, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.dft = Dft2d(width, height, fixed=fixed)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.dft(x)
        out = self.pool(out)
        return out


class ComparisonBlockDctHybrid(nn.Module):
    """
    Mimicks the discrete cosine II 2D hybrid block but with randomly initialized weights for the spectral transform.
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
                 groups_conv=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.lin = DctII2dComparison(width, height)
        self._1x1 = nn.Conv2d(2 * out_channels, out_channels, 1, groups=2)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.lin(x)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
        out = self._1x1(out)
        return out


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
                 groups_conv=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.cos = iDctII2d(width, height, fixed=fixed)
        self._1x1 = nn.Conv2d(2 * out_channels, out_channels, 1, groups=2)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.cos(x)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
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
                 groups_conv=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.cos = iDctII2d(width, height, fixed=fixed)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.cos(x)
        out = self.pool(out)
        return out


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
                 groups_conv=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.ifft = iDft2d(width, height, fixed=fixed)
        self._1x1 = nn.Conv2d(3 * out_channels, out_channels, 1, groups=2)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.ifft(x)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
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
                 groups_conv=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.dft = iDft2d(width, height, fixed=fixed)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.dft(x)
        x = self.pool(x)
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
                 groups_conv=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self._1x1 = nn.Conv2d(2 * out_channels, out_channels, 1, groups=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self._1x1(out)
        return out


class DctII2dComparison(nn.Module):
    """
    DctII2d with random weights.
    """

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.weights_1 = nn.Parameter(self._create_weight_tensor())
        self.weights_2 = nn.Parameter(self._create_weight_tensor())

    def _create_weight_tensor(self, ):
        w = torch.Tensor(self.width, self.height)
        stdv = 1. / math.sqrt(w.size(1))
        w.uniform_(-stdv, stdv)
        return w

    def forward(self, input):
        x = F.linear(torch.transpose(input, -2, -1), self.weights_1)
        return F.linear(torch.transpose(x, -2, -1), self.weights_2)


class DftComparison(nn.Module):

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.weights_1 = nn.Parameter(self._create_weight_tensor())
        self.weights_2 = nn.Parameter(self._create_weight_tensor())
        self.weights_3 = nn.Parameter(self._create_weight_tensor())
        self.weights_4 = nn.Parameter(self._create_weight_tensor())

    def _create_weight_tensor(self, ):
        w = torch.Tensor(self.width, self.height)
        stdv = 1. / math.sqrt(w.size(1))
        w.uniform_(-stdv, stdv)
        return w

    def forward(self, input):
        x = F.linear(torch.transpose(input, -2, -1), self.weights_1)
        return F.linear(torch.transpose(x, -2, -1), self.weights_2)


class DctIIComparisonBlock(nn.Module):
    """
    Defines a block mimicking the DctII2D Blocks but with randomly initialized weights (Xavier uniform like).
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
                 groups_conv=2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.lin = DctII2dComparison(width, height)
        self._1x1 = nn.Conv2d(2 * out_channels, out_channels, 1, groups=2)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.lin(x)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
        out = self._1x1(out)
        return out





