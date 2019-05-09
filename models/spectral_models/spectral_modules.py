import torch.nn as nn
from spectral import Dft2d, iDft2d, DctII2d, iDctII2d


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

class DiscreteCosine2dConvBlock(nn.Module):

    def __init__(self,
                 width,
                 height,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 fixed=False,
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.cos = DctII2d(width, height, fixed=fixed)

    def forward(self, x):
        x = self.conv(x)
        x = self.cos(x)
        return x


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
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.dft = Dft2d(width, height, fixed=fixed)

    def forward(self, x):
        x = self.conv(x)
        x = self.dft(x)
        return x


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
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.cos = iDctII2d(width, height, fixed=fixed)

    def forward(self, x):
        x = self.conv(x)
        x = self.cos(x)
        return x

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
                 groups_conv=1,
                 padding=1):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups_conv, padding=padding)
        self.dft = iDft2d(width, height, fixed=fixed)

    def forward(self, x):
        x = self.conv(x)
        x = self.dft(x)
        return x


