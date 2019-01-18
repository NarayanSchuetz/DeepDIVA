"""
 Created by Narayan Schuetz at 07/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


from .parts import *
import torch.nn as nn
from .registry import Model


@Model
class PureConv_150x150(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 ocl1=32,  # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (127, 127)

        self.network = nn.Sequential(
            ConvBlock(None, None, input_channels, ocl1, 7, fixed=fixed, padding=3, stride=2),
            nn.LeakyReLU(),
            ConvBlock(None, None, ocl1, ocl1*2, 5, fixed=fixed, padding=2, stride=2),
            nn.LeakyReLU(),
            ConvBlock(None, None, ocl1*2, ocl1*4, 3, fixed=fixed, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=32, stride=1),
            Flatten(),
            nn.Linear(ocl1*4, output_channels)
        )

    def forward(self, x):
        return self.network(x)


@Model
class PureConv_32x32(nn.Module):

    def __init__(self,
                 output_channels=10,
                 input_channels=3,
                 fixed=True,
                 ocl1=16, # output channels layer 1
                 **kwargs):

        super().__init__()

        self.expected_input_size = (32, 32)

        self.network = nn.Sequential(
            ConvBlock(32, 32, input_channels, ocl1, 3, fixed=fixed, padding=1, stride=1),
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
