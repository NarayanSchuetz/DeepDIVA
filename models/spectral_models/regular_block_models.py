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
class CosineBidirectional_150x150_Fixed(nn.Module):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """

    def __init__(self,
                 output_channels=8,
                 input_channels=3,
                 fixed=True,
                 ocl1=32,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlock(64, 64, input_channels, ocl1, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlock(32, 32, ocl1, ocl1*2, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class CosineBidirectional_150x150_Unfixed(CosineBidirectional_150x150_Fixed):
    """
    Network performing cosine transforms.
    150x150 -> DCTII -> iDCTII -> DCTII
    """
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=27, ** kwargs)

@Model
class FourierBidirectional_150x150_Fixed(nn.Module):
    """
    Network performing cosine transforms.
    150x150 -> Dft -> iDft -> Dft
    """

    def __init__(self,
                 output_channels=8,
                 input_channels=3,
                 fixed=True,
                 ocl1=32,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(64, 64, input_channels, ocl1, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlock(32, 32, ocl1*2, ocl1*2, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(32, 32, ocl1*2, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class FourierBidirectional_150x150_Unfixed(FourierBidirectional_150x150_Fixed):
    """
    Network performing cosine transforms.
    150x150 -> Dft -> iDft -> Dft
    """
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=20, ** kwargs)

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
                 ocl1=16,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlock(32, 32, input_channels, ocl1, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteCosine2dConvBlock(32, 32, ocl1, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteCosine2dConvBlock(32, 32, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class CosineBidirectional_32x32_Unfixed(CosineBidirectional_32x32_Fixed):
    """
    Network performing cosine transforms.
    32x32 -> DCTII -> iDCTII -> DCTII
    """
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=14, ** kwargs)


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
                 ocl1=16,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(32, 32, input_channels, ocl1, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            InverseDiscreteFourier2dConvBlock(32, 32, ocl1*2, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            DiscreteFourier2dConvBlock(32, 32, ocl1*2, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class FourierBidirectional_32x32_Unfixed(FourierBidirectional_32x32_Fixed):
    """
    Network performing cosine transforms.
    32x32 -> Dft -> iDft -> Dft
    """

    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=12, ** kwargs)

# ----------------------------------------------------------------------------------------------------------------------
# Only First Block Spectral Models
# ----------------------------------------------------------------------------------------------------------------------


@Model
class FirstCosine_150x150_Fixed(nn.Module):

    def __init__(self,
                 output_channels=8,
                 input_channels=3,
                 fixed=True,
                 ocl1=32,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlock(64, 64, input_channels, ocl1, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1, ocl1*2, 5, fixed=fixed, padding=2, stride=2),
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
class FirstCosine_150x150_Unfixed(FirstCosine_150x150_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=29, ** kwargs)


@Model
class FirstFourier_150x150_Fixed(nn.Module):

    def __init__(self,
                 output_channels=8,
                 input_channels=3,
                 fixed=True,
                 ocl1=32,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(64, 64, input_channels, ocl1, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1*2, ocl1*2, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1*2, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class FirstFourier_150x150_Unfixed(FirstFourier_150x150_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=23, ** kwargs)


@Model
class FirstCosine_32x32_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 ocl1=16,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteCosine2dConvBlock(32, 32, input_channels, ocl1, 3, fixed=fixed, padding=1, stride=1),
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
class FirstCosine_32x32_Unfixed(FirstCosine_32x32_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=15, **kwargs)


@Model
class FirstFourier_32x32_Fixed(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 ocl1=16,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            DiscreteFourier2dConvBlock(32, 32, input_channels, ocl1, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            ConvBlock(32, 32, ocl1*2, ocl1*2, 3, fixed=fixed, padding=1, stride=1),
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
class FirstFourier_32x32_Unfixed(FirstFourier_32x32_Fixed):
    def __init__(self, output_channels=10, input_channels=3, fixed=False, **kwargs):
        super().__init__(output_channels=output_channels, input_channels=input_channels, fixed=fixed, ocl1=15,**kwargs)