from .spectral_modules import *
import torch.nn as nn
from models.registry import Model


@Model
class BaselineConv(nn.Module):

    def __init__(self, output_channels, in_channels=3, ocl1=32,  # output channels layer 1
                 **kwargs):
        super().__init__()

        self.expected_input_size = (149, 149)
        self.features = []

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, ocl1, kernel_size=8, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(ocl1, ocl1 * 2, kernel_size=5, stride=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(ocl1 * 2, ocl1 * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=1),
            Flatten(),
            nn.Linear(ocl1 * 4, output_channels)
        )

    def forward(self, x):
        self.features = self.encoder(x)
        x = self.classifier(self.features)
        return x

