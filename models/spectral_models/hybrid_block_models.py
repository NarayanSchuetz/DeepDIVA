"""
 Created by Narayan Schuetz at 07/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


from .parts import *
import torch.nn as nn
from .registry import Model


# ----------------------------------------------------------------------------------------------------------------------
# Bidirectional Models
# ----------------------------------------------------------------------------------------------------------------------


@Model
class HybridFourierBidirectional_150x150_Fixed(nn.Module):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=8,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybridMaxPool(127, 127, input_channels, 48, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlockHybridMaxPool(64, 64, 48, 96, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlockHybrid(32, 32, 96, 204, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(204, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFourierBidirectional_150x150_Unfixed(nn.Module):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=8,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybridMaxPool(127, 127, input_channels, 40, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlockHybridMaxPool(64, 64, 40, 86, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlockHybrid(32, 32, 86, 180, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(180, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridCosineBidirectional_150x150_Fixed(nn.Module):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=8,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybridMaxPool(127, 127, input_channels, 46, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlockHybridMaxPool(64, 64, 46, 100, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlockHybrid(32, 32, 100, 212, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(212, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridCosineBidirectional_150x150_Unfixed(nn.Module):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=8,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybridMaxPool(127, 127, input_channels, 46, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlockHybridMaxPool(64, 64, 46, 96, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlockHybrid(32, 32, 96, 192, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(192, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridCosineBidirectional_32x32_Fixed(nn.Module):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybrid(32, 32, input_channels, 55, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlockHybrid(32, 32, 55, 110, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlockHybrid(32, 32, 110, 220, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(220, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridCosineBidirectional_32x32_Unfixed(nn.Module):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybrid(32, 32, input_channels, 55, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlockHybrid(32, 32, 55, 110, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlockHybrid(32, 32, 110, 220, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(220, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFourierBidirectional_32x32_Fixed(nn.Module):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybrid(32, 32, input_channels, 52, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlockHybrid(32, 32, 52, 106, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlockHybrid(32, 32, 106, 220, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(220, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFourierBidirectional_32x32_Unfixed(nn.Module):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybrid(32, 32, input_channels, 52, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlockHybrid(32, 32, 52, 106, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlockHybrid(32, 32, 106, 212, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(212, output_channels)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------∂-------------------------------------------------------------------------
# Only First Block Spectral Models
# ----------------------------------------------------------------------------------------------------------------------


@Model
class HybridFirstCosine_150x150_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybridMaxPool(127, 127, input_channels, 56, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(64, 64, 56, 112, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 112, 224, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(224, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstCosine_150x150_Unfixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybridMaxPool(127, 127, input_channels, 54, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(64, 64, 54, 106, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 106, 216, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(216, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstFourier_150x150_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybridMaxPool(127, 127, input_channels, 56, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(64, 64, 56, 112, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 112, 224, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(224, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstFourier_150x150_Unfixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybridMaxPool(127, 127, input_channels, 51, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(64, 64, 51, 102, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 102, 204, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(204, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstCosine_32x32_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybrid(32, 32, input_channels, 64, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 64, 128, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 128, 256, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(256, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstCosine_32x32_Unfixed(HybridFirstCosine_32x32_Fixed):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)



@Model
class HybridFirstFourier_32x32_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybrid(32, 32, input_channels, 64, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 64, 128, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 128, 256, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(256, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstFourier_32x32_Unfixed(HybridFirstFourier_32x32_Fixed):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)

class BaseModel_32x32(nn.Module):
    """Abstract Model for input with size 32x32"""

    def __init__(self,
                 first_block,
                 second_block,
                 third_block,
                 output_channels=10,
                 input_channels=3,
                 fixed=None,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            first_block(28, 28, input_channels, 16, 5, kernel_size_pooling=2, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            second_block(12, 12, 16, 32, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            third_block(4, 4, 32, 32, 3, kernel_size_pooling=1, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(512, output_channels),
        )

    def forward(self, x):
        return self.network(x)

@Model
class Test(BaseModel_32x32):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            MyDiscreteCosine2dConvBlockHybrid,
            MyInverseDiscreteCosine2dConvBlockHybrid,
            MyDiscreteCosine2dConvBlockHybrid,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs)

    def forward(self, x):
        return self.network(x)

class MyDiscreteCosine2dConvBlockHybrid(nn.Module):
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


class MyInverseDiscreteCosine2dConvBlockHybrid(nn.Module):
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
