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

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            first_block(144, 144, input_channels, 16, 7, kernel_size_pooling=4, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            second_block(32, 32, 16, 32, 5, kernel_size_pooling=4, fixed=fixed),
            nn.LeakyReLU(),
            third_block(6, 6, 32, 64, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class BaseModel_150x150_Dft_Unfixed(nn.Module):
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

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            first_block(144, 144, input_channels, 36, 7, kernel_size_pooling=4, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            second_block(32, 32, 36, 64, 5, kernel_size_pooling=4, fixed=fixed),
            nn.LeakyReLU(),
            third_block(6, 6, 64, 64, 3, kernel_size_pooling=2, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(576, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class BaseModel_150x150_Dft_Fixed(nn.Module):
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

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            first_block(144, 144, input_channels, 48, 7, kernel_size_pooling=4, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            second_block(32, 32, 48, 96, 5, kernel_size_pooling=4, fixed=fixed),
            nn.LeakyReLU(),
            third_block(6, 6, 96, 164, 3, kernel_size_pooling=3, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(656, output_channels),
        )

    def forward(self, x):
        return self.network(x)


class BaseModel_150x150_Dft_Only_Fixed(nn.Module):
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

        self.expected_input_size = (150, 150)

        self.network = nn.Sequential(
            first_block(144, 144, input_channels, 96, 7, kernel_size_pooling=4, groups_conv=1, fixed=fixed),
            nn.LeakyReLU(),
            second_block(32, 32, 96, 148, 5, kernel_size_pooling=4, fixed=fixed),
            nn.LeakyReLU(),
            third_block(6, 6, 150, 148, 3, kernel_size_pooling=3, fixed=fixed),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(656, output_channels),
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
class CosineBidirectional_150x150_Fixed(BaseModel_150x150):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlock,
            InverseDiscreteCosine2dConvBlock,
            DiscreteCosine2dConvBlock,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class CosineBidirectional_150x150_Unfixed(BaseModel_150x150):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlock,
            InverseDiscreteCosine2dConvBlock,
            DiscreteCosine2dConvBlock,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class FourierBidirectional_150x150_Fixed(BaseModel_150x150_Dft_Fixed):
    """
    Network performing cosine transforms.
    150x150 -> Dft -> iDft -> Dft
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlock,
            InverseDiscreteFourier2dConvBlock,
            DiscreteFourier2dConvBlock,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class FourierBidirectional_150x150_Unfixed(BaseModel_150x150_Dft_Unfixed):
    """
    Network performing cosine transforms.
    150x150 -> Dft -> iDft -> Dft
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlock,
            InverseDiscreteFourier2dConvBlock,
            DiscreteFourier2dConvBlock,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class CosineBidirectional_32x32_Fixed(nn.Module):
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
            DiscreteCosine2dConvBlock(32, 32, input_channels, 64, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlock(32, 32, 64, 128, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(32, 32, 128, 256, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(218, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class CosineBidirectional_32x32_Unfixed(CosineBidirectional_32x32_Fixed):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)



@Model
class FourierBidirectional_32x32_Fixed(nn.Module):
    """
    Network performing cosine transforms.
    32x32 -> Dft -> iDft -> Dft
    """

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(32, 32, input_channels, 64, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlock(32, 32, 64, 128, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(32, 32, 128, 258, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(218, output_channels)
        )

    def forward(self, x):
        return self.network(x)

@Model
class FourierBidirectional_32x32_Unfixed(FourierBidirectional_32x32_Fixed):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# Only First Block Spectral Models
# ----------------------------------------------------------------------------------------------------------------------


@Model
class FirstCosine_150x150_Fixed(BaseModel_150x150):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlock,
            ConvBlock,
            ConvBlock,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class FirstCosine_150x150_Unfixed(BaseModel_150x150):
    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlock,
            ConvBlock,
            ConvBlock,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class FirstFourier_150x150_Fixed(BaseModel_150x150):
    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlock,
            ConvBlock,
            ConvBlock,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class FirstFourier_150x150_Unfixed(BaseModel_150x150):
    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlock,
            ConvBlock,
            ConvBlock,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class FirstCosine_32x32_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlock(32, 32, input_channels, 64, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 64, 128, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 128, 258, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(218, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class FirstCosine_32x32_Unfixed(FirstCosine_32x32_Fixed):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)


@Model
class FirstFourier_32x32_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(32, 32, input_channels, 64, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 64, 128, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, 128, 258, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(218, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class FirstFourier_32x32_Unfixed(FirstFourier_32x32_Fixed):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=False,
                 **kwargs):

        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, **kwargs)