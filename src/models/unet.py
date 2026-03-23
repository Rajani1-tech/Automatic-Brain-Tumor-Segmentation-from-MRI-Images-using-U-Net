# src/models/unet_model.py
import torch
import torch.nn as nn
from configs.config import Config


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetModel(nn.Module):
    def __init__(self, in_channels=Config.IMG_CHANNELS, out_channels=Config.NUM_CLASSES):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec6 = ConvBlock(512 + 256, 256)

        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec7 = ConvBlock(256 + 128, 128)

        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec8 = ConvBlock(128 + 64, 64)

        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec9 = ConvBlock(64 + 32, 32)

        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        c5 = self.bottleneck(self.pool(c4))

        x = self.dec6(torch.cat([self.up6(c5), c4], dim=1))
        x = self.dec7(torch.cat([self.up7(x), c3], dim=1))
        x = self.dec8(torch.cat([self.up8(x), c2], dim=1))
        x = self.dec9(torch.cat([self.up9(x), c1], dim=1))

        return torch.sigmoid(self.out_conv(x))