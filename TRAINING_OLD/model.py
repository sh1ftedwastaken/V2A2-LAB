import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two 3x3 conv layers with ReLU; optional 2D dropout at the end."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        ]
        if dropout_p and dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightUNet(nn.Module):
    """
    Encoder: three downsampling stages with avg-pooling, keeping pre-pool features as skips.
    Decoder: three transposed-conv upsampling stages with skip concatenation.

    Input:  (N, 3, 320, 240)
    Output: (N, num_classes, 320, 240)
    """

    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        # Encoder
        self.stem = DoubleConv(3, 16, dropout_p=0.0)
        self.enc1 = DoubleConv(16, 32, dropout_p=0.0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv(32, 64, dropout_p=dropout_p)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv(64, 128, dropout_p=dropout_p)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 128, dropout_p=dropout_p)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128 + 64, 64, dropout_p=dropout_p)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 32, 32, dropout_p=0.0)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(32 + 16, 16, dropout_p=0.0)

        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        skip1 = self.enc1(x0)
        x = self.pool1(skip1)

        skip2 = self.enc2(x)
        x = self.pool2(skip2)

        skip3 = self.enc3(x)
        x = self.pool3(skip3)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec3(x)

        return self.head(x)


__all__ = ["DoubleConv", "LightUNet"]