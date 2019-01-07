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
class HybridCosineBidirectional_32x32_Fixed(BaseModel_32x32):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
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
class HybridCosineBidirectional_32x32_Unfixed(BaseModel_32x32):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
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


# ----------------------------------------------------------------------------------------------------------------------
# Pure Spectral Models
# ----------------------------------------------------------------------------------------------------------------------


@Model
class HybridCosine_150x150_Fixed(BaseModel_150x150):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> DCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridCosine_150x150_Unfixed(BaseModel_150x150):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> DCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridFourier_150x150_Fixed(BaseModel_150x150):
    """
    Network performing cosine transforms.
    150x150 -> DFT -> DFT -> DFT
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlockHybrid,
            DiscreteFourier2dConvBlockHybrid,
            DiscreteFourier2dConvBlockHybrid,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridFourier_150x150_Unfixed(BaseModel_150x150):
    """
    Network performing cosine transforms.
    150x150 -> DFT -> DFT -> DFT
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlockHybrid,
            DiscreteFourier2dConvBlockHybrid,
            DiscreteFourier2dConvBlockHybrid,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridCosine_32x32_Fixed(BaseModel_32x32):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> DCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridCosine_32x32_Unfixed(BaseModel_32x32):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> DCTII -> DCTII
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            DiscreteCosine2dConvBlockHybrid,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridFourier_32x32_Fixed(BaseModel_32x32):
    """
    Network performing cosine transforms.
    32x32 -> DFT -> DFT -> DFT
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlockHybrid,
            DiscreteFourier2dConvBlockHybrid,
            DiscreteFourier2dConvBlockHybrid,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridFourier_32x32_Unfixed(BaseModel_32x32):
    """
    Network performing cosine transforms.
    32x32 -> DFT -> DFT -> DFT
    """

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteFourier2dConvBlockHybrid,
            DiscreteFourier2dConvBlockHybrid,
            DiscreteFourier2dConvBlockHybrid,
            fixed=False,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


# ----------------------------------------------------------------------------------------------------------------------
# Only First Block Spectral Models
# ----------------------------------------------------------------------------------------------------------------------


@Model
class HybridFirstCosine_150x150_Fixed(BaseModel_150x150):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            ConvBlock,
            ConvBlock,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


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
class HybridFirstCosine_32x32_Fixed(BaseModel_32x32):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            ConvBlock,
            ConvBlock,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridFirstCosine_32x32_Unfixed(BaseModel_32x32):

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
class HybridFirstFourier_32x32_Fixed(BaseModel_32x32):

    def __init__(self, output_channels=10, input_channels=3, **kwargs):
        super().__init__(
            DiscreteCosine2dConvBlockHybrid,
            ConvBlock,
            ConvBlock,
            fixed=True,
            output_channels=output_channels,
            input_channels=input_channels,
            **kwargs
        )


@Model
class HybridFirstFourier_32x32_Unfixed(BaseModel_32x32):

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
