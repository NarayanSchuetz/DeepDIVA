"""
 Created by Narayan Schuetz at 07/01/2019
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


import math
import torch.nn as nn
import torch
import torch.nn.functional as F

from spectral import Dft2d, iDft2d, DctII2d, iDctII2d


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

    def __init__(self, new_width, new_height):
        super().__init__()

        self.width = new_width
        self.height = new_height

    def forward(self, x):
        return x[:, :, :self.height, :self.width].clone()


class DctII2dComparison(nn.Module):
    """
    DctII2d random weights.
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
        self._1x1 = nn.Conv2d(2 * out_channels, out_channels, 1, groups=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.cos(x)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
        out = self._1x1(out)
        return out


"""
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
"""

class DctBlockDirect(nn.Module):

    def __init__(self,
                 width,
                 height,
                 fixed=False):

        super().__init__()
        self.cos = DctII2d(width, height, fixed=fixed)
        self.icos = iDctII2d(width//3,  width//3, fixed=fixed)
        self.spectral_pool = Dct2dPooling( width//3,  width//3)

    def forward(self, x):
        out = self.cos(x)
        out = self.spectral_pool(out)
        out = self.icos(out)
        return out


class DctBlockConv(nn.Module):

    def __init__(self,
                 width,
                 height,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 fixed=False):

        super().__init__()
        self.cos = DctII2d(width, height, fixed=fixed)
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size)
        self.icos = iDctII2d(width//2, width//2, fixed=fixed)
        self.spectral_pool = Dct2dPooling(width//2, width//2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.cos(x)
        out = self.conv(out)
        out = self.relu(out)
        out = self.spectral_pool(out)
        out = self.icos(out)
        return out


class DiscreteCosine2dConvBlockSpectralPooling(nn.Module):

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
        self._1x1 = nn.Conv2d(2 * out_channels, out_channels, 1, groups=2)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)
        self.spectral_pool = Dct2dPooling(14, 14)

    def forward(self, x):
        x = self.conv(x)
        out = self.cos(x)
        x = self.pool(x)
        out = self.spectral_pool(out)
        out = torch.cat((out, x), 1)
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


class ComparisonBlock(nn.Module):

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
        self._1x1 = nn.Conv2d(2 * out_channels, out_channels, 1, groups=2)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)

    def forward(self, x):
        x = self.conv(x)
        out = self.cos(x)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
        out = self._1x1(out)
        return out


"""
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
"""


class ConvBlock(nn.Module):

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


#-----------------------------------------------------------------------------------------------------------------------
# NETWORKS
#-----------------------------------------------------------------------------------------------------------------------


class CosineBidirectional_150x150_Fixed(nn.Module):
    """
    Block performing cosine transforms.
        150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlock(144, 144, input_channels, 16, 7, kernel_size_pooling=4, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlock(32, 32, 16, 32, 5, kernel_size_pooling=4, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(6, 6, 32, 64, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class CosineBidirectional_150x150_Unfixed(CosineBidirectional_150x150_Fixed):

    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class FourierBidirectional_150x150_Fixed(nn.Module):
    """
    Block performing cosine transforms.
        150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(144, 144, input_channels, 16, 7, kernel_size_pooling=4, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlock(32, 32, 16, 32, 5, kernel_size_pooling=4, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(6, 6, 32, 64, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class FourierBidirectional_150x150_Unfixed(FourierBidirectional_150x150_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class CosineBidirectional_30x30_Fixed(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockSpectralPooling(28, 28, input_channels, 16, 5, kernel_size_pooling=2, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlock(12, 12, 16, 32, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(4, 4, 32, 32, 3, kernel_size_pooling=1, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(512, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class CosineBidirectional_30x30_Unfixed(CosineBidirectional_30x30_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class FourierBidirectional_30x30_Fixed(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(28, 28, input_channels, 16, 5, kernel_size_pooling=2, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlock(12, 12, 16, 32, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(4, 4, 32, 32, 3, kernel_size_pooling=1, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(512, output_channels)
        )

    def forward(self, x):
        return self.network(x)


class FourierBidirectional_30x30_Unfixed(FourierBidirectional_30x30_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class Cosine_150x150_Fixed(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlock(144, 144, input_channels, 16, 7, kernel_size_pooling=4, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(32, 32, 16, 32, 5, kernel_size_pooling=4, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(6, 6, 32, 64, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels)
        )

    def forward(self, x):
        return self.network(x)


class Cosine_150x150_Unfixed(Cosine_150x150_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class Fourier_150x150_Fixed(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(144, 144, input_channels, 16, 7, kernel_size_pooling=4, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(32, 32, 16, 32, 5, kernel_size_pooling=4, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(6, 6, 32, 64, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels)
        )

    def forward(self, x):
        return self.network(x)


class Fourier_150x150_Unfixed(Fourier_150x150_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class Cosine_30x30_Fixed(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlock(28, 28, input_channels, 16, 5, kernel_size_pooling=2, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(12, 12, 16, 32, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(4, 4, 32, 32, 3, kernel_size_pooling=1, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(512, output_channels)
        )

    def forward(self, x):
        return self.network(x)


class Cosine_30x30_Unfixed(Cosine_30x30_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class Fourier_30x30_Fixed(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(28, 28, input_channels, 16, 5, kernel_size_pooling=2, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(12, 12, 16, 32, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(4, 4, 32, 32, 3, kernel_size_pooling=1, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(512, output_channels)
        )

    def forward(self, x):
        return self.network(x)


class Fourier_30x30_Unfixed(Fourier_30x30_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class CosineShallow_30x30_Fixed(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()
        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlock(28, 28, input_channels, 24, 5, kernel_size_pooling=2, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlock(12, 12, 24, 24, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(864, output_channels)
        )

    def forward(self, x):
        return self.network(x)


class CosineShallow_30x30_Unfixed(CosineShallow_30x30_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class LinearShallow_30x30(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__()
        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            ComparisonBlock(28, 28, input_channels, 24, 5, kernel_size_pooling=2, groups_conv=1),
            nn.LeakyReLU(),
            ComparisonBlock(12, 12, 24, 24, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(864, output_channels)
        )

    def forward(self, x):
        return self.network(x)


class Linear_150x150(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__()

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            ComparisonBlock(144, 144, input_channels, 16, 7, kernel_size_pooling=4, groups_conv=1),
            nn.LeakyReLU(),
            ComparisonBlock(32, 32, 16, 32, 5, kernel_size_pooling=4),
            nn.LeakyReLU(),
            ComparisonBlock(6, 6, 32, 64, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class PureConv_150x150(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__()

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            ConvBlock(144, 144, input_channels, 16, 7, kernel_size_pooling=4, groups_conv=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 16, 32, 5, kernel_size_pooling=4),
            nn.LeakyReLU(),
            ConvBlock(6, 6, 32, 64, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels),
        )

    def forward(self, x):
        return self.network(x)



class Linear_30x30(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            ComparisonBlock(28, 28, input_channels, 16, 5, kernel_size_pooling=2, groups_conv=1),
            nn.LeakyReLU(),
            ComparisonBlock(12, 12, 16, 32, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            ComparisonBlock(4, 4, 32, 32, 3, kernel_size_pooling=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(512, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class PureConv_30x30(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            ConvBlock(28, 28, input_channels, 16, 5, kernel_size_pooling=2, groups_conv=1),
            nn.LeakyReLU(),
            ConvBlock(12, 12, 16, 32, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            ConvBlock(4, 4, 32, 32, 3, kernel_size_pooling=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(512, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class FirstSpectralConv_150x150_Fixed(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(144, 144, input_channels, 16, 7, kernel_size_pooling=4, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 16, 32, 5, kernel_size_pooling=4),
            nn.LeakyReLU(),
            ConvBlock(6, 6, 32, 64, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class FirstSpectralConv_150x150_Unfixed(FirstSpectralConv_150x150_Fixed):

    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class FirstSpectralConv_30x30_Fixed(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(28, 28, input_channels, 16, 5, kernel_size_pooling=2, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            ConvBlock(12, 12, 16, 32, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            ConvBlock(4, 4 , 32, 32, 3, kernel_size_pooling=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(512, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class FirstSpectralConv_30x30_Unfixed(FirstSpectralConv_30x30_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


class DctPoolingNetwork(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, fixed=True, **kwargs):
        super().__init__()

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            DctBlockDirect(150, 150, fixed),
            nn.Conv2d(input_channels, 6, 5),
            nn.LeakyReLU(),
            DctBlockDirect(46, 46, fixed),
            nn.Conv2d(6, 12, 3),
            nn.LeakyReLU(),
            DctBlockDirect(13, 13, fixed),
            nn.Conv2d(12, 24, 2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(216, output_channels)
        )

    def forward(self, x):
        return self.network(x)




class SpectralBlockSimple(nn.Module):

    def __init__(self, nrows, ncols, n_linear_layers=2, fixed=True):
        super().__init__()
        self._n_linear_layer = n_linear_layers
        self._nrows = nrows
        self._ncols = ncols

        self._transform = DctII2d(nrows, ncols, fixed=fixed)
        self._linear = self._create_linear_layers(nrows*ncols)
        self._inverse_transform = iDctII2d(nrows, ncols, fixed=fixed)

        self._relu = nn.LeakyReLU()

    def forward(self, x):

        x = self._transform(x)
        #x = self._relu(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self._linear(x)
        x = x.view(x.shape[0], x.shape[1], self._nrows, self._ncols)
        x = self._inverse_transform(x)
        #x = self._relu(x)

        return x

    def _create_linear_layers(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )


class SpectralBlockFft(nn.Module):

    def __init__(self, nrows, ncols, n_linear_layers=2, fixed=True):
        super().__init__()
        self._n_linear_layer = n_linear_layers
        self._nrows = nrows
        self._ncols = ncols

        self._transform = Fft2d(nrows, ncols, fixed=fixed)
        self._linear = self._create_linear_layers(nrows*2*ncols)
        self._inverse_transform = iFft2d(nrows, ncols, fixed=fixed, mode="amp")

        self._relu = nn.LeakyReLU()

    def forward(self, x):

        x = self._transform(x)
        x = self._relu(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self._linear(x)
        x = x.view(x.shape[0], x.shape[1], self._nrows, self._ncols*2)
        x = self._inverse_transform(x)

        #x = self._relu(x)

        return x[:,:,:,:x.shape[-1]//2]

    def _create_linear_layers(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU()
        )


class ComparisonBlock(nn.Module):

    def __init__(self, nrows, ncols, n_linear_layers=2, fixed=True):
        super().__init__()
        self._n_linear_layer = n_linear_layers
        self._nrows = nrows
        self._ncols = ncols

        self._linear = self._create_linear_layers(nrows*ncols)

        self._relu = nn.LeakyReLU()

    def forward(self, x):

        x = x.view(x.shape[0], x.shape[1], -1)
        x = self._linear(x)
        x = x.view(x.shape[0], x.shape[1], self._nrows, self._ncols)

        return x

    def _create_linear_layers(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU()
        )


class Spectral2dBasic(nn.Module):
    """
    Basic CNN extended with 2D FFT layer

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super().__init__()

        self.expected_input_size = (32, 32)

        # Zeroth layer
        self.spectral_block = SpectralBlockSimple(32, 32)
        self.spectral_block_2 = SpectralBlockSimple(10, 10)


        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=5, stride=3),
            nn.LeakyReLU()
        )
        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2),
            nn.LeakyReLU()
        )
        # Third layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(288, 10)
        )
        self.fl = Flatten()
        self.rel = nn.LeakyReLU()
        self.con2d = nn.Conv2d(3, 24, kernel_size=5, stride=3)

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        #x = self.spectral_block(x)
        #x = self.spectral_block(x)
        #identity = x.clone()

        identity = x.clone()
        x = self.spectral_block(x)
        x = self.conv1(torch.cat((x, identity), 1))
        identity = x.clone()
        x = self.spectral_block_2(x)
        x = self.conv2(torch.cat((x, identity), 1))
        x = self.conv3(x)
        x = self.fc(x)

        return x


class Spectral2dFft(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super().__init__()

        self.expected_input_size = (32, 32)

        # Zeroth layer
        self.spectral_block = SpectralBlockSimple(32, 32)
        self.spectral_block_2 = SpectralBlockFft(10, 10)


        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=3),
            nn.LeakyReLU()
        )
        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2),
            nn.LeakyReLU()
        )
        # Third layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(288, 10)
        )
        self.fl = Flatten()
        self.rel = nn.LeakyReLU()
        self.con2d = nn.Conv2d(3, 24, kernel_size=5, stride=3)

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        #x = self.spectral_block(x)
        #x = self.spectral_block(x)
        #identity = x.clone()
        x = self.conv1(x)
        identity = x.clone()
        x = self.spectral_block_2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)

        return x


class SkipIntermediateDiscreteCosine2dConvBlock(nn.Module):

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
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv)
        self.cos = DctII2d(width, height, fixed=fixed)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)
        self._1x1 = nn.Conv2d(2*out_channels, 12, 1, groups=2)
        self._1x1_ng = nn.Conv2d(2*out_channels, 12, 1, groups=1)

    def forward(self, x):
        #x = self.relu(x)
        x = self.conv(x)
        out = self.cos(x)
        out = self.relu(out)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
        skip = self._1x1_ng(out)
        out = self._1x1(out)
        return out, skip


class IntermediateDiscreteCosine2dConvBlock(nn.Module):

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
        self._1x1 = nn.Conv2d(2*out_channels, out_channels, 1, groups=2)

    def forward(self, x):
        x = self.conv(x)
        out = self.cos(x)
        out = torch.cat((out, x), 1)
        out = self.pool(out)
        out = self._1x1(out)
        return out


class ComparisonIntermediateDiscreteCosine2dConvBlock(nn.Module):

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
        self.cos = nn.Linear(width*height, width*height, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pooling if kernel_size_pooling is not None else kernel_size)
        self._1x1 = nn.Conv2d(2*out_channels, out_channels, 1, groups=2)
        self.batchnorm = nn.BatchNorm2d(12)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.conv(x)
        out = self.cos(x.view(x.shape[0], x.shape[1], -1))
        out = self.cos(out)
        out = out.view(out.shape[0], out.shape[1], int(math.sqrt(out.shape[-1])), int(math.sqrt(out.shape[-1])))
        out = torch.cat((out, x), 1)
        out = self.pool(out)
        out = self._1x1(out)
        return out


class SpectralConv1(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__()
        print(output_channels)

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            IntermediateDiscreteCosine2dConvBlock(28, 28, input_channels, 24, 5, kernel_size_pooling=2, groups_conv=1),
            nn.LeakyReLU(),
            IntermediateDiscreteCosine2dConvBlock(12, 12, 24, 24, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(864, output_channels)
        )

    def forward(self, x):
        return self.network(x)


class SpectralConv1FullSize(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__()
        print(output_channels)

        self.expected_input_size = (150, 150)


        self.network = nn.Sequential(

            IntermediateDiscreteCosine2dConvBlock(144, 144, input_channels, 16, 7, kernel_size_pooling=4, groups_conv=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            IntermediateDiscreteCosine2dConvBlock(32, 32, 16, 32, 5, kernel_size_pooling=4),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            IntermediateDiscreteCosine2dConvBlock(6, 6, 32, 64, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            Flatten(),
            nn.Linear(576, output_channels)
        )

    def forward(self, x):
        return self.network(x)


class SpectralConv1SkipConnections(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__()
        print(output_channels)

        self.expected_input_size = (150, 150)

        self.relu = nn.ReLU()
        self.block0 = nn.Conv2d(3, 12, 7)
        self.block1 = SkipIntermediateDiscreteCosine2dConvBlock(140, 140, 12, 12, 5, kernel_size_pooling=1)
        self.block2 = SkipIntermediateDiscreteCosine2dConvBlock(138, 138, 12, 12, 3, kernel_size_pooling=1)
        #self.block3 = SkipIntermediateDiscreteCosine2dConvBlock(136, 136, 12, 12, 3, kernel_size_pooling=1)
        self.blockn = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(12,8, 3),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            Flatten(),
            nn.Linear(512, output_channels)
        )

    def forward(self, x):
        skip_connections = []
        x = self.block0(x)
        x = self.relu(x)
        out, skip = self.block1(x)
        skip_size = 138
        skip_connections.append(skip[:, :, -skip_size:, -skip_size:])
        out, skip = self.block2(out)
        skip_connections.append(skip[:, :, -skip_size:, -skip_size:])
        skip = torch.stack(skip_connections)
        out = torch.sum(skip, dim=0)
        out = self.blockn(out)
        return out


class SpectralConv1Comparison(nn.Module):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__()
        print(output_channels)

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            ComparisonIntermediateDiscreteCosine2dConvBlock(28, 28, input_channels, 24, 5, kernel_size_pooling=2, groups_conv=1),
            nn.LeakyReLU(),
            ComparisonIntermediateDiscreteCosine2dConvBlock(12, 12, 24, 24, 3, kernel_size_pooling=2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(864, output_channels)
        )

    def forward(self, x):
        return self.network(x)
