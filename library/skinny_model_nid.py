from library.skinny_segments import InitConv, ConvDown, DeconvUp, FinalConv, AdditionalLayers
import torch.nn as nn


# Skinny architecture variant without inception modules and dense blocks
class Skinny(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Skinny, self).__init__()

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

    def forward(self, x):
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
