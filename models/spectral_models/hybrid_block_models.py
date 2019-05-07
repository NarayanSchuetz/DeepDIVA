"""
 Created by Narayan Schuetz at 07/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


from .parts import *
import torch.nn as nn
from models.registry import Model


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
                 ocl1=30,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybridMaxPool(127, 127, input_channels, ocl1, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlockHybridMaxPool(64, 64, ocl1, ocl1*2, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlockHybrid(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFourierBidirectional_150x150_Unfixed(HybridFourierBidirectional_150x150_Fixed):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=20, **kwargs)


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
                 ocl1=30,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybridMaxPool(127, 127, input_channels, ocl1, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlockHybridMaxPool(64, 64, ocl1, ocl1*2, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlockHybrid(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridCosineBidirectional_150x150_Unfixed(HybridCosineBidirectional_150x150_Fixed):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=25, **kwargs)


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
                 ocl1=14, # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybrid(32, 32, input_channels, ocl1, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlockHybrid(32, 32, ocl1, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlockHybrid(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridCosineBidirectional_32x32_Unfixed(HybridCosineBidirectional_32x32_Fixed):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
    """
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=12, **kwargs)

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
                 ocl1=14,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybrid(32, 32, input_channels, ocl1, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlockHybrid(32, 32, ocl1, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlockHybrid(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFourierBidirectional_32x32_Unfixed(HybridFourierBidirectional_32x32_Fixed):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
    """
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=10, **kwargs)


# ---------------------------------------------∂-------------------------------------------------------------------------
# Only First Block Spectral Models
# ----------------------------------------------------------------------------------------------------------------------
@Model
class HybridFirstCosine_150x150_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 ocl1=32,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybridMaxPool(127, 127, input_channels, ocl1, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(64, 64, ocl1, ocl1*2, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstCosine_150x150_Unfixed(HybridFirstCosine_150x150_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=29, **kwargs)


@Model
class HybridFirstFourier_150x150_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 ocl1=32,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybridMaxPool(127, 127, input_channels, ocl1, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(64, 64, ocl1, ocl1*2, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstFourier_150x150_Unfixed(HybridFirstFourier_150x150_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=23, **kwargs)


@Model
class HybridFirstCosine_32x32_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 ocl1=16,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlockHybrid(32, 32, input_channels, ocl1, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstCosine_32x32_Unfixed(HybridFirstCosine_32x32_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=15, **kwargs)


@Model
class HybridFirstFourier_32x32_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 ocl1=16,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlockHybrid(32, 32, input_channels, ocl1, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class HybridFirstFourier_32x32_Unfixed(HybridFirstFourier_32x32_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=15, **kwargs)

