#!/usr/bin/env python3
"""
U-Net6Ch loader for cpu_v4_best.pth.
6-channel input, attention gates at dec3/dec4, no DSConv.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.gn = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=True)
        self.psi = nn.Conv2d(F_int, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        attn = self.relu(self.W_g(g) + self.W_x(x))
        attn = self.sigmoid(self.psi(attn))
        return x * attn


class UNet6Ch(nn.Module):
    """6-channel U-Net with attention gates, matching cpu_v4_best.pth exactly."""
    def __init__(self, in_channels=6, base_filters=64):
        super().__init__()
        self.enc_block1 = nn.Sequential(ConvBlock(in_channels, base_filters), ConvBlock(base_filters, base_filters))
        self.pool1 = nn.MaxPool2d(2)
        self.enc_block2 = nn.Sequential(ConvBlock(base_filters, base_filters*2), ConvBlock(base_filters*2, base_filters*2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc_block3 = nn.Sequential(ConvBlock(base_filters*2, base_filters*4), ConvBlock(base_filters*4, base_filters*4))
        self.pool3 = nn.MaxPool2d(2)
        self.enc_block4 = nn.Sequential(ConvBlock(base_filters*4, base_filters*8), ConvBlock(base_filters*8, base_filters*8))
        self.pool4 = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(ConvBlock(base_filters*8, base_filters*16), ConvBlock(base_filters*16, base_filters*16))
        self.upconv4 = nn.ConvTranspose2d(base_filters*16, base_filters*8, 2, stride=2)
        self.attn4 = AttentionGate(base_filters*8, base_filters*8, base_filters*4)
        self.dec_block4 = nn.Sequential(ConvBlock(base_filters*16, base_filters*8), ConvBlock(base_filters*8, base_filters*8))
        self.upconv3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 2, stride=2)
        self.attn3 = AttentionGate(base_filters*4, base_filters*4, base_filters*2)
        self.dec_block3 = nn.Sequential(ConvBlock(base_filters*8, base_filters*4), ConvBlock(base_filters*4, base_filters*4))
        self.upconv2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.dec_block2 = nn.Sequential(ConvBlock(base_filters*4, base_filters*2), ConvBlock(base_filters*2, base_filters*2))
        self.upconv1 = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.dec_block1 = nn.Sequential(ConvBlock(base_filters*2, base_filters), ConvBlock(base_filters, base_filters))
        self.final_conv = nn.Conv2d(base_filters, 1, 1)

    def forward(self, x):
        enc1 = self.enc_block1(x)
        enc2 = self.enc_block2(self.pool1(enc1))
        enc3 = self.enc_block3(self.pool2(enc2))
        enc4 = self.enc_block4(self.pool3(enc3))
        bridge = self.bridge(self.pool4(enc4))
        up4 = self.upconv4(bridge)
        enc4_gated = self.attn4(g=up4, x=enc4)
        dec4 = self.dec_block4(torch.cat([up4, enc4_gated], dim=1))
        up3 = self.upconv3(dec4)
        enc3_gated = self.attn3(g=up3, x=enc3)
        dec3 = self.dec_block3(torch.cat([up3, enc3_gated], dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec_block2(torch.cat([up2, enc2], dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec_block1(torch.cat([up1, enc1], dim=1))
        return self.final_conv(dec1)
