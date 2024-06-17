from torch import nn


class DepthWiseConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "valid",
    ):
        super(DepthWiseConvolution, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=stride,
            padding="valid",
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x
