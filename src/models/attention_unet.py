import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):

    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.theta_x = nn.Conv2d(x_channels,    inter_channels, 1)
        self.phi_g   = nn.Conv2d(g_channels,     inter_channels, 1)
        self.psi     = nn.Conv2d(inter_channels, 1, 1)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        attn = self.sigmoid(self.psi(self.relu(self.theta_x(x) + self.phi_g(g))))
        return x * attn


class AttentionUNetModel(nn.Module):

    def __init__(self, mode="binary"):
        super().__init__()
        assert mode in ("binary", "multiclass")
        self.mode    = mode

        # binary  → 1 channel  (single grayscale modality)
        # multiclass → 4 channels (T1, T1CE, T2, FLAIR stacked)
        in_channels  = Config.IMG_CHANNELS if mode == "binary" else Config.MULTICLASS_IN_CHANNELS
        out_channels = 1 if mode == "binary" else Config.NUM_CLASSES

        self.enc1       = ConvBlock(in_channels, 32)
        self.enc2       = ConvBlock(32, 64)
        self.enc3       = ConvBlock(64, 128)
        self.enc4       = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.pool       = nn.MaxPool2d(2)

        self.att4       = AttentionGate(256, 512, 256)
        self.att3       = AttentionGate(128, 256, 128)
        self.att2       = AttentionGate(64,  128, 64)
        self.att1       = AttentionGate(32,  64,  32)

        self.up6        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec6       = ConvBlock(512 + 256, 256)
        self.up7        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec7       = ConvBlock(256 + 128, 128)
        self.up8        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec8       = ConvBlock(128 + 64, 64)
        self.up9        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec9       = ConvBlock(64 + 32, 32)

        self.out_conv   = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        c5 = self.bottleneck(self.pool(c4))

        x  = self.dec6(torch.cat([self.up6(c5), self.att4(c4, c5)], dim=1))
        x  = self.dec7(torch.cat([self.up7(x),  self.att3(c3, x)],  dim=1))
        x  = self.dec8(torch.cat([self.up8(x),  self.att2(c2, x)],  dim=1))
        x  = self.dec9(torch.cat([self.up9(x),  self.att1(c1, x)],  dim=1))

        logits = self.out_conv(x)
        return torch.sigmoid(logits) if self.mode == "binary" else logits