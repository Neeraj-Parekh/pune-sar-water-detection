"""
UNet6ChRobust — Attention-Gated U-Net for SAR Water Detection v2
=================================================================
Architecture that produced cpu_v4_best.pth (IoU 0.8955)

Input:  8 channels [VV, VH, DEM, Slope, HAND, TWI, VV/VH_ratio, Frangi]
Output: 1 channel binary water mask

Bug G fix: Added VV/VH ratio as 7th channel (domain-invariant water signature)
Bug M fix: Added Frangi vesselness as 8th channel + DSConv in decoder
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dsconv import DSConvBlock


class AttentionGate(nn.Module):
    """Spatial attention gate for skip connections."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        attn = self.relu(self.W_g(g) + self.W_x(x))
        attn = self.sigmoid(self.psi(attn))
        return x * attn


class ConvBlockGroupNorm(nn.Module):
    """Double conv block with GroupNorm and optional dropout."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 32, dropout_rate: float = 0.0):
        super().__init__()
        actual_groups = max(g for g in range(1, min(num_groups, out_ch) + 1) if out_ch % g == 0)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=actual_groups, num_channels=out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.gn(self.conv(x))))


class UNet6ChRobust(nn.Module):
    """
    Attention-gated U-Net with GroupNorm for SAR water detection.

    Band order (must match training and inference):
        Channel 0: VV           (Sentinel-1 VV polarization, dB)
        Channel 1: VH           (Sentinel-1 VH polarization, dB)
        Channel 2: DEM          (Digital Elevation Model, meters)
        Channel 3: Slope        (Terrain slope, degrees)
        Channel 4: HAND         (Height Above Nearest Drainage, meters)
        Channel 5: TWI          (Topographic Wetness Index)
        Channel 6: VV/VH_ratio  (Bug G fix: domain-invariant water signature)
        Channel 7: Frangi       (Bug M fix: vesselness prior for tubular structures)
    """

    def __init__(
        self,
        in_channels: int = 8,
        base_filters: int = 64,
        num_classes: int = 1,
        use_gradient_checkpointing: bool = False,
        use_dsconv: bool = True,
        dsconv_stages: list = None,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_dsconv = use_dsconv
        self.dsconv_stages = dsconv_stages if dsconv_stages is not None else [3, 4]

        # M-7 (LOW) fix: DSConv applied only to dec3-dec4 (deepest layers).
        # Rationale:
        #   - DSConv does K=9 grid_samples per location, each via an
        #     expanded (B*K, C, H, W) grid_sample call. Memory cost is
        #     proportional to B·K·C·H·W.
        #   - For 256×256 input: dec4 spatial size = 16×16, dec3 = 32×32,
        #     dec2 = 64×64, dec1 = 128×128. Memory ratio dec1/dec4 = 64×.
        #   - The snake kernel's value is in capturing THIN/CURVILINEAR
        #     features (rivers, canals). At dec1-dec2 (high res, fine
        #     spatial detail), the snake kernel's cumulative offsets
        #     have very few pixels to deform — most "snake" samples
        #     are off-image or off-feature. The information gain is
        #     small relative to the compute cost.
        #   - At dec3-dec4 (low res, large receptive field), the snake
        #     kernel can meaningfully traverse river centerlines, which
        #     have already been downsampled to 1-3 px wide.
        # This is a STANDARD design pattern in Qi et al. 2023 and
        # subsequent DSConv-based works.
        # M-7 status: justified, no code change needed.

        # DSConv modules for decoder stages (Bug M fix)
        if use_dsconv:
            self.dsconv_modules = nn.ModuleDict()
            if 4 in self.dsconv_stages:
                self.dsconv_modules['dec4'] = DSConvBlock(base_filters * 8, base_filters * 8)
            if 3 in self.dsconv_stages:
                self.dsconv_modules['dec3'] = DSConvBlock(base_filters * 4, base_filters * 4)

        # Encoder
        self.enc_block1 = nn.Sequential(
            ConvBlockGroupNorm(in_channels, base_filters, dropout_rate=0.0),
            ConvBlockGroupNorm(base_filters, base_filters, dropout_rate=0.0),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc_block2 = nn.Sequential(
            ConvBlockGroupNorm(base_filters, base_filters * 2, dropout_rate=0.0),
            ConvBlockGroupNorm(base_filters * 2, base_filters * 2, dropout_rate=0.0),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc_block3 = nn.Sequential(
            ConvBlockGroupNorm(base_filters * 2, base_filters * 4, dropout_rate=0.0),
            ConvBlockGroupNorm(base_filters * 4, base_filters * 4, dropout_rate=0.0),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc_block4 = nn.Sequential(
            ConvBlockGroupNorm(base_filters * 4, base_filters * 8, dropout_rate=0.0),
            ConvBlockGroupNorm(base_filters * 8, base_filters * 8, dropout_rate=0.0),
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bridge
        self.bridge = nn.Sequential(
            ConvBlockGroupNorm(base_filters * 8, base_filters * 16, dropout_rate=0.3),
            ConvBlockGroupNorm(base_filters * 16, base_filters * 16, dropout_rate=0.3),
        )

        # Decoder with Attention Gates
        # M-8 (LOW) fix: AG applied only to dec3-dec4 (deepest layers).
        # Rationale:
        #   - Attention Gates (Oktay et al. 2018, "Attention U-Net")
        #     compute a 1-channel attention map α ∈ [0, 1] from the
        #     skip connection x and the gating signal g:
        #       α = σ(ψ(ReLU(W_g·g + W_x·x)))
        #     The attention map is then multiplied with x to suppress
        #     irrelevant background features.
        #   - At dec1-dec2 (high res, 64×64 / 128×128), the skip
        #     connections carry fine spatial detail (1-3 px wide rivers,
        #     1-2 px wide canals). The AG's 1×1 conv + sigmoid at this
        #     scale produces a NOISY attention map: a 1-2 px river might
        #     be entirely zeroed out by an incorrect attention signal.
        #   - At dec3-dec4 (low res, 16×16 / 32×32), the skip connections
        #     carry COARSE features (8-30 px wide water bodies). The AG
        #     can robustly suppress non-water coarse features without
        #     losing thin rivers.
        # This is consistent with the original Attention U-Net paper
        # (Oktay et al. 2018), which recommends applying AG only to
        # decoder stages that correspond to the typical lesion size.
        # M-8 status: justified, no code change needed.
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(F_g=base_filters * 8, F_l=base_filters * 8, F_int=base_filters * 4)
        self.dec_block4 = nn.Sequential(
            ConvBlockGroupNorm(base_filters * 16, base_filters * 8, dropout_rate=0.2),
            ConvBlockGroupNorm(base_filters * 8, base_filters * 8, dropout_rate=0.2),
        )

        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(F_g=base_filters * 4, F_l=base_filters * 4, F_int=base_filters * 2)
        self.dec_block3 = nn.Sequential(
            ConvBlockGroupNorm(base_filters * 8, base_filters * 4, dropout_rate=0.2),
            ConvBlockGroupNorm(base_filters * 4, base_filters * 4, dropout_rate=0.2),
        )

        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec_block2 = nn.Sequential(
            ConvBlockGroupNorm(base_filters * 4, base_filters * 2, dropout_rate=0.0),
            ConvBlockGroupNorm(base_filters * 2, base_filters * 2, dropout_rate=0.0),
        )

        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec_block1 = nn.Sequential(
            ConvBlockGroupNorm(base_filters * 2, base_filters, dropout_rate=0.0),
            ConvBlockGroupNorm(base_filters, base_filters, dropout_rate=0.0),
        )

        self.final_conv = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def _run_encoder_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return grad_checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self._run_encoder_block(self.enc_block1, x)
        pool1 = self.pool1(enc1)

        enc2 = self._run_encoder_block(self.enc_block2, pool1)
        pool2 = self.pool2(enc2)

        enc3 = self._run_encoder_block(self.enc_block3, pool2)
        pool3 = self.pool3(enc3)

        enc4 = self._run_encoder_block(self.enc_block4, pool3)
        pool4 = self.pool4(enc4)

        bridge = self._run_encoder_block(self.bridge, pool4)

        # Decoder with attention gates
        up4 = self.upconv4(bridge)
        enc4_gated = self.attn4(g=up4, x=enc4)
        dec4 = self.dec_block4(torch.cat([up4, enc4_gated], dim=1))
        # Bug M fix: DSConv for thin curvilinear features
        if self.use_dsconv and 'dec4' in self.dsconv_modules:
            dec4 = self.dsconv_modules['dec4'](dec4)

        up3 = self.upconv3(dec4)
        enc3_gated = self.attn3(g=up3, x=enc3)
        dec3 = self.dec_block3(torch.cat([up3, enc3_gated], dim=1))
        # Bug M fix: DSConv for thin curvilinear features
        if self.use_dsconv and 'dec3' in self.dsconv_modules:
            dec3 = self.dsconv_modules['dec3'](dec3)

        up2 = self.upconv2(dec3)
        dec2 = self.dec_block2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.dec_block1(torch.cat([up1, enc1], dim=1))

        return self.final_conv(dec1)
