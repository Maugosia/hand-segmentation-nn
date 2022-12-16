from .segments import InitConv, ConvDown, DeconvUp, FinalConv, AdditionalLayers, InceptionModule, ConvInception, FirstUpscaling, Upscaling, Final

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkinnyInception(nn.Module):
    """
    Skinny architecture variant with inception modules.
    """

    def __init__(self, n_channels: int, n_classes: int):
        super(SkinnyInception, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = (ConvInception(self.n_channels, 20))
        self.down2 = (ConvInception(20, 40))
        self.down3 = (ConvInception(40, 80))
        self.down4 = (ConvInception(80, 160))
        self.down5 = (ConvInception(160, 320))

        self.up1 = (FirstUpscaling(320))
        self.up2 = (Upscaling(640))
        self.up3 = (Upscaling(320))
        self.up4 = (Upscaling(160))
        self.up5 = (Upscaling(80))

        self.final = (Final(40, self.n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x2-x6 receive max-pooled inputs
        x1 = self.down1(x)
        x2 = self.down2(F.max_pool2d(x1, 2))
        x3 = self.down3(F.max_pool2d(x2, 2))
        x4 = self.down4(F.max_pool2d(x3, 2))
        x5 = self.down5(F.max_pool2d(x4, 2))

        x6 = self.up1(F.max_pool2d(x5, 2), x5)
        x6 = self.up2(x6, x4)
        x6 = self.up3(x6, x3)
        x6 = self.up4(x6, x2)
        x6 = self.up5(x6, x1)

        x6 = self.final(x6)

        return x6


class SkinnyBasic(nn.Module):
    """
    Skinny architecture variant without inception modules.
    """

    def __init__(self, n_channels: int, n_classes: int):
        super(SkinnyBasic, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.init = (InitConv(self.n_channels, 15))

        self.conv_down1 = (ConvDown(15, 30))
        self.conv_down2 = (ConvDown(30, 60))
        self.conv_down3 = (ConvDown(60, 120))
        self.conv_down4 = (ConvDown(120, 240))

        self.additional = (AdditionalLayers())

        self.deconv_up1 = (DeconvUp(240, 480))
        self.deconv_up2 = (DeconvUp(480, 240))
        self.deconv_up3 = (DeconvUp(240, 120))
        self.deconv_up4 = (DeconvUp(120, 60))
        self.deconv_up5 = (DeconvUp(60, 30))

        self.final = (FinalConv(30, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.init(x)
        x2 = self.conv_down1(x1)
        x3 = self.conv_down2(x2)
        x4 = self.conv_down3(x3)
        x5 = self.conv_down4(x4)
        x6 = self.additional(x5)

        x7 = self.deconv_up1(x6, x5)
        x8 = self.deconv_up2(x7, x4)
        x9 = self.deconv_up3(x8, x3)
        x10 = self.deconv_up4(x9, x2)
        x11 = self.deconv_up5(x10, x1)

        output = self.final(x11)

        return output
