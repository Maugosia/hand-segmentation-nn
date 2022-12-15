import torch.nn as nn
import torch.nn.functional as F
import torch

padding_mode = "same"


# first part - double convolution


class InitConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sequential_stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.sequential_stack(x)


# downscaling - double convolution with max pooling
class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sequential_stack = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.sequential_stack(x)


# upscaling - double convolution with convolution transpose and concatenation
class DeconvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_channels = out_channels // 2

        self.sequential_stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=mid_channels,
                               stride=2, kernel_size=3),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.sequential_stack(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)


# final module with several convolution layers finished with sigmoid
class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()

        mid_channels = in_channels // 2

        self.sequential_stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels // 2,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels // 2, out_channels=out_channels,
                      padding=padding_mode, kernel_size=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequential_stack(x)


class AdditionalLayers(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential_stack = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.sequential_stack(x)