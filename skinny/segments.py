import torch
import torch.nn as nn
import torch.nn.functional as F

PADDING_MODE = "same"


class InceptionModule(nn.Module):
    """
    Inception module - novelty in comparison to the basic Skinny.
    Replaces one of the standard 3x3 convolution layers in the double convolution block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        mid_channels = out_channels // 4

        self.first_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.second_path = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.third_path = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fourth_path = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=5, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.first_path(x)
        x2 = self.second_path(x)
        x3 = self.third_path(x)
        x4 = self.fourth_path(x)

        # the first path has been downscaled and has to be padded
        # to make concatenation possible
        x_axis_difference = x2.size()[3] - x1.size()[3]
        y_axis_difference = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [x_axis_difference // 2,
                        x_axis_difference - x_axis_difference // 2,
                        y_axis_difference // 2,
                        y_axis_difference - y_axis_difference // 2])

        return torch.cat([x1, x2, x3, x4], dim=1)


class ConvInception(nn.Module):
    """
    Convolution block which precedes the pooling layer.
    Separation from the pooling layer is necessery due
    to the need of saving non-pooled output for the skip
    connections.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.pre_downscale = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            InceptionModule(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pre_downscale(x)


class FirstUpscaling(nn.Module):
    """
    First upscaling block after contracting path. Contains
    additonal convolution layer before the inception module
    compared to the regular upscaling block.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        mid_channels = in_channels * 2

        self.upscaling = nn.Sequential(
            ConvInception(in_channels, mid_channels),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=in_channels,
                               stride=2, kernel_size=3),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upscaling(x1)

        # x2 comes from the contracting path and has
        # different image size compared to x1 which
        # comes from the expansive path, so it
        # has to be padded to match x1's size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2])

        return torch.cat([x2, x1], dim=1)


class Upscaling(nn.Module):
    """
    Upscaling block terminated with concatenation
    due to the skip connection between contracting
    and expansive paths.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        mid_channels = [in_channels // 2, in_channels // 4]

        self.upscaling = nn.Sequential(
            InceptionModule(in_channels, mid_channels[0]),
            nn.ConvTranspose2d(in_channels=mid_channels[0], out_channels=mid_channels[1],
                               stride=2, kernel_size=3),
            nn.Conv2d(in_channels=mid_channels[1], out_channels=mid_channels[1],
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upscaling(x1)

        # x2 comes from the contracting path and has
        # different image size compared to x1 which
        # comes from the expansive path, so it
        # has to be padded to match x1's size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2])

        return torch.cat([x2, x1], dim=1)


class Final(nn.Module):
    """
    Final block of the network terminated with sigmoid function.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        mid_channels = in_channels // 2

        self.final = nn.Sequential(
            InceptionModule(40, 20),
            nn.Conv2d(in_channels=20, out_channels=20,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=20, out_channels=10,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=10, out_channels=out_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(x)


############# LEGACY VERSION - Skinny without inception module #######################

# first part - double convolution
class InitConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sequential_stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
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
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
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
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=mid_channels,
                               stride=2, kernel_size=3),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.sequential_stack(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)


# final module with several convolution layers finished with sigmoid
class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()

        mid_channels = in_channels // 2

        self.sequential_stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels // 2,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels // 2, out_channels=out_channels,
                      padding=PADDING_MODE, kernel_size=3, bias=False),
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
