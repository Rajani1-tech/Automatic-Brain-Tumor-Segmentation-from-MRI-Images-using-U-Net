# src/models/attention_unet.py
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


class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.theta_x = nn.Conv2d(x_channels, inter_channels, 1, padding=0)
        self.phi_g = nn.Conv2d(g_channels, inter_channels, 1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # g may be spatially smaller — upsample to match x
        if g.shape[2:] != x.shape[2:]:
            g = nn.functional.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        theta = self.theta_x(x)
        phi = self.phi_g(g)
        attn = self.sigmoid(self.psi(self.relu(theta + phi)))
        return x * attn


class AttentionUNetModel(nn.Module):
    def __init__(self, in_channels=Config.IMG_CHANNELS, out_channels=Config.NUM_CLASSES):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Attention gates
        self.att4 = AttentionGate(256, 512, 256)
        self.att3 = AttentionGate(128, 256, 128)
        self.att2 = AttentionGate(64, 128, 64)
        self.att1 = AttentionGate(32, 64, 32)

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

        u6 = self.up6(c5)
        x = self.dec6(torch.cat([u6, self.att4(c4, c5)], dim=1))

        u7 = self.up7(x)
        x = self.dec7(torch.cat([u7, self.att3(c3, x)], dim=1))

        u8 = self.up8(x)
        x = self.dec8(torch.cat([u8, self.att2(c2, x)], dim=1))

        u9 = self.up9(x)
        x = self.dec9(torch.cat([u9, self.att1(c1, x)], dim=1))

        return torch.sigmoid(self.out_conv(x))