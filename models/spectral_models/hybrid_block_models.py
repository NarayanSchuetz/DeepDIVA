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
# Base Models
# ----------------------------------------------------------------------------------------------------------------------


class BaseModel_150x150(nn.Module):
    """Abstract Model for input with size 150x150"""

    def __init__(self,
                 first_block,
                 second_block,
                 third_block,
                 output_channels=10,
                 input_channels=3,
                 fixed=None,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            first_block(127, 127, input_channels, 54, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            second_block(64, 64, 16, 32, 5, kernel_size_pooling=4, fixed=fixed),
            nn.LeakyReLU(),
            third_block(6, 6, 32, 64, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels),
        )

    def forward(self, x):
        return self.network(x)


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


# ----------------------------------------------------------------------------------------------------------------------
# Bidirectional Models
# ----------------------------------------------------------------------------------------------------------------------


@Model
class HybridCosineBidirectional_150x150_Fixed(BaseModel_150x150):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            InverseDiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridCosineBidirectional_150x150_Unfixed(BaseModel_150x150):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            InverseDiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


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
            nn.Linear(212, output_channels)
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
            DiscreteCosine2dConvBlockHybrid(127, 127, input_channels, 56, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(64, 64, 56, 112, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 112, 224, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(500, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstCosine_150x150_Unfixed(BaseModel_150x150):
    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            ConvBlock,
            ConvBlock,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridFirstFourier_150x150_Fixed(BaseModel_150x150):
    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlockHybrid,
            ConvBlock,
            ConvBlock,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridFirstFourier_150x150_Unfixed(BaseModel_150x150):
    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlockHybrid,
            ConvBlock,
            ConvBlock,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


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
