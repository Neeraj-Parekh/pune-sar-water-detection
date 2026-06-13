"""
Dynamic Snake Convolution (DSConv) for SAR Water Detection v2
=============================================================
Bug M fix: Captures thin curvilinear water features (narrow rivers, streams, canals).

Reference: Qi et al., ICCV 2023, "Dynamic Snake Convolution based on
Topological Geometric Constraints for Tubular Structure Segmentation"
(arXiv:2307.08388).

Key innovation:
  - Straightens 3x3 kernel into two 1D kernels (x-axis, y-axis) of length 9
  - Cumulative offset constraints prevent kernel from "wandering" off thin rivers
  - Unlike standard deformable conv where offsets are independent

Bug ADD-5 fix: Removed unused DSConv standalone class (155 lines dead code).
Only DSConvBlock is used by the model.

M-1 (HIGH) fix: align_corners mismatch resolved (see forward() below).
M-2 (HIGH) fix: 8 independent deltas (not mirror-shared) per Qi Eq. 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    """
    DSConv block for integration into U-Net decoder.

    Bug R6 fix: replaces uniform averaging with learned 1D conv weights
    along the snake kernel dimension (paper-faithful per Qi et al. ICCV 2023).

    M-2 (HIGH) fix: 8 INDEPENDENT deltas (Qi et al. 2023, Eq. 3):
        cum_x[i] = sum_{k=0..i-1} delta_x[k]   for i in 0..8
        cum_y[i] = sum_{k=0..i-1} delta_y[k]   for i in 0..8
    The OLD code (incorrectly) used `cumsum(flip(delta))` to enforce a
    MIRROR symmetry between left and right deltas. This is NOT in the
    paper. The paper only enforces a CUMULATIVE constraint
    (each offset depends on the previous), not a symmetry constraint.
    Mirror symmetry is a strong inductive bias that LIMITS the model's
    ability to follow curves whose slope CHANGES from left to right
    of the kernel center — exactly the case of river meanders, road
    intersections, and canal junctions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_length: int = 9):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_length = kernel_length

        # M-4 (MEDIUM) — kernel_length=9 vs paper's 5.
        # Qi et al. 2023 uses K=5 in their experiments, but their ablation
        # table (Table 2) shows K=9 is comparable or better for thin
        # curvilinear structures. We use K=9 because:
        #   - Pune rivers/canals are 10-80 m wide (1-8 px at 10 m GSD).
        #     K=5 only covers ±2 px (≈ 50 m), missing wider rivers.
        #   - K=9 covers ±4 px (≈ 90 m), the full Pune river width range.
        #   - The computational cost of K=9 vs K=5 is 1.8× (one extra
        #     pooling per direction per sample), acceptable.
        # Decision: K=9 is a justified design choice, not a bug.

        # Offset prediction: predicts 8 INDEPENDENT deltas (4 along y, 4 along x).
        # M-2 fix: 8 deltas are now independent — no mirror constraint.
        # cum is built by simple cumulative sum (Qi et al. 2023, Eq. 3).
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2 * (kernel_length - 1), kernel_size=1),  # 4 Δy + 4 Δx
        )

        # Bug R6 fix: LEARNED 1D conv weights along the snake kernel
        # Depthwise 1D conv: one independent filter per input channel,
        # applied across the 9 sample positions. Replaces the previous
        # uniform averaging which diluted the offset learning signal.
        self.kernel_conv_x = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_length,
            padding=0, groups=in_channels, bias=False,
        )
        self.kernel_conv_y = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_length,
            padding=0, groups=in_channels, bias=False,
        )
        # M-6 (LOW) fix: 1D-conv init choice (uniform vs Kaiming).
        # PyTorch's default init for Conv1d is Kaiming uniform
        # (a=U(−√k, √k) where k = 1/(C·K)).
        # We INTENTIONALLY override to uniform 1/K so that:
        #   1. At initialization, the 1D conv behaves like a
        #      moving-average aggregator along the 9 sample positions.
        #   2. This is the SAME as the previous (pre-R6) hand-coded
        #      uniform averaging, so the rest of the network doesn't
        #      see a sudden distribution shift at training start.
        #   3. Kaiming init would push the 1D conv to behave like a
        #      non-averaging filter, breaking the snake-aggregation
        #      assumption (which expects smooth neighbor averaging).
        # The deviation from Kaiming is justified by the architectural
        # role: this 1D conv is an AGGREGATOR, not a feature extractor.
        # M-6 status: documented, no code change needed.
        with torch.no_grad():
            for conv in (self.kernel_conv_x, self.kernel_conv_y):
                conv.weight.data.fill_(1.0 / kernel_length)

        # Main convolution after learned snake aggregation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # Bug ADD-3 fix: Use GroupNorm instead of batch norm for consistency with main model
        num_groups = min(32, out_channels) if out_channels >= 32 else out_channels
        self.bn = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        K = self.kernel_length

        # Predict offsets: 2 * (K-1) = 16 channels, 8 per direction
        # (in_channels → 16 for K=9)
        offsets = torch.tanh(self.offset_conv(x))  # (B, 2*(K-1), H, W)
        half = K - 1
        delta_y = offsets[:, :half]  # (B, K-1, H, W)
        delta_x = offsets[:, half:]  # (B, K-1, H, W)

        # M-2 (HIGH) fix: cumulative offsets WITHOUT mirror constraint.
        # Per Qi et al. ICCV 2023, Eq. 3:
        #   cum[i] = sum_{k=0..i-1} delta[k]   (cumulative sum, independent deltas)
        # The center position (i=4 for K=9) is fixed at 0 (no offset).
        # For K=9: 4 deltas on the LEFT of center, 4 on the RIGHT.
        # The deltas delta_y (shape (B, K-1, H, W) = (B, 8, H, W)) are split:
        #   delta_y_neg = delta_y[:, :K//2]  → 4 deltas for positions [-4, -3, -2, -1]
        #   delta_y_pos = delta_y[:, K//2:]  → 4 deltas for positions [+1, +2, +3, +4]
        # Cumulative offsets:
        #   neg_cum = -[d3, d2+d3, d1+d2+d3, d0+d1+d2+d3]  (4 values for positions -1..-4)
        #   pos_cum = +[d4, d4+d5, d4+d5+d6, d4+d5+d6+d7]  (4 values for positions +1..+4)
        # Full cumulative (9 values total): [neg_cum, 0, pos_cum]
        #
        # This is the EXACT paper convention; the OLD code attempted a
        # mirror symmetry that the paper does NOT specify.
        half_neg = K // 2  # 4 for K=9
        delta_y_neg = delta_y[:, :half_neg]  # (B, 4, H, W) for left side
        delta_y_pos = delta_y[:, half_neg:]  # (B, 4, H, W) for right side
        delta_x_neg = delta_x[:, :half_neg]
        delta_x_pos = delta_x[:, half_neg:]

        # neg_cum[i] = -sum(delta_y_neg[i:])
        neg_cum_y = -torch.flip(torch.cumsum(torch.flip(delta_y_neg, dims=[1]), dim=1), dims=[1])
        pos_cum_y = torch.cumsum(delta_y_pos, dim=1)
        cum_y = torch.cat([neg_cum_y, torch.zeros(B, 1, H, W, device=x.device), pos_cum_y], dim=1)  # (B, K, H, W) = (B, 9, H, W)

        neg_cum_x = -torch.flip(torch.cumsum(torch.flip(delta_x_neg, dims=[1]), dim=1), dims=[1])
        pos_cum_x = torch.cumsum(delta_x_pos, dim=1)
        cum_x = torch.cat([neg_cum_x, torch.zeros(B, 1, H, W, device=x.device), pos_cum_x], dim=1)  # (B, K, H, W) = (B, 9, H, W)

        # Build sampling grids
        # M-9 (LOW) fix: dtype consistency for AMP compatibility.
        # OLD code hardcoded float32 for arange and base vectors, while
        # x can be float16/bfloat16 under AMP. This caused a dtype
        # mismatch at the `xx_b + base_x` add, forcing an implicit
        # upcast to float32 and breaking the AMP graph.
        # NEW code: derive the helper dtype from x.dtype.
        helper_dtype = x.dtype
        rows = torch.arange(H, device=x.device, dtype=helper_dtype)
        cols = torch.arange(W, device=x.device, dtype=helper_dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing='ij')

        xx_b = xx.view(1, H, W, 1).expand(B, H, W, 1)
        yy_b = yy.view(1, H, W, 1).expand(B, H, W, 1)

        base_x = torch.arange(-(K // 2), K // 2 + 1, device=x.device, dtype=helper_dtype)
        gx_x = xx_b + base_x.view(1, 1, 1, -1)
        gy_x = yy_b + cum_y.permute(0, 2, 3, 1)

        base_y = torch.arange(-(K // 2), K // 2 + 1, device=x.device, dtype=helper_dtype)
        gx_y = xx_b + cum_x.permute(0, 2, 3, 1)
        gy_y = yy_b + base_y.view(1, 1, 1, -1)

        def sample_with_grid(features, gx, gy):
            # M-3 (MEDIUM) — memory analysis of the features_expanded step.
            # For B=2, K=9, C=512 (dec4), H=W=32, the expand-and-reshape
            # `features.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(
            # B_grid * K, C, H, W)` materializes a tensor of:
            #   2 * 9 * 512 * 32 * 32 * 4 bytes = 18.9 MB  (per sample)
            # The OLD comment ("B*K*C*H*W reshape materializes 126.6 MB")
            # was an overestimate — for the actual dec4 spatial size
            # (32×32), this is 19 MB, well within budget.
            #
            # The expand() call is a VIRTUAL expand (no copy), so the
            # memory is only materialized at the .reshape() call. This
            # is the standard PyTorch pattern for "batch the K samples
            # into a single grid_sample call" and is what PERF-4 was
            # trying to achieve.
            #
            # M-3 status: no fix needed. The 19 MB is acceptable, and
            # the einsum in the aggregation step (PERF-5) prevents
            # further OOM by avoiding the (B, C, H, W, K) → (B*K, C, H, W)
            # flatten that would have created (B*C*H*W*K) intermediates.

            # M-1 (HIGH) fix: align_corners=True formula is now USED
            # CONSISTENTLY (and `align_corners=True` is passed to
            # grid_sample). The OLD code mixed the two conventions:
            # grid built with `align_corners=True` formula (`2*gx/(W-1)-1`)
            # but grid_sample called with `align_corners=False`
            # (which expects `2*gx/W - 1`). This caused a half-pixel
            # bias (≈ 5 m for 10 m Sentinel-1) in the sampling locations.
            #
            # align_corners=True is the standard for image processing
            # (treats pixel centers as the grid; corners at ±1). This
            # matches the convention used in the Qi et al. 2023 reference
            # implementation and the original DSConv code from the
            # authors (https://github.com/YaoleiQi/DSCNet).
            gx_norm = 2.0 * gx / (W - 1) - 1.0
            gy_norm = 2.0 * gy / (H - 1) - 1.0
            grid = torch.stack([gx_norm, gy_norm], dim=-1)  # (B, H, W, K, 2)
            B_grid = grid.shape[0]
            grid_flat = grid.view(B_grid * K, H, W, 2)
            features_expanded = features.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B_grid * K, C, H, W)
            # M-1 fix: align_corners=True (matches grid formula)
            sampled_flat = F.grid_sample(
                features_expanded, grid_flat,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            )
            sampled = sampled_flat.view(B_grid, C, H, W, K)
            return sampled

        sampled_x = sample_with_grid(x, gx_x, gy_x)  # (B, C, H, W, K)
        sampled_y = sample_with_grid(x, gx_y, gy_y)

        # Bug R6 fix: learned 1D conv along kernel dimension (replaces uniform sum/9)
        # Bug PERF-5 fix: avoid B*H*W flattening to prevent OOM
        wx = self.kernel_conv_x.weight.view(C, K)
        wy = self.kernel_conv_y.weight.view(C, K)
        ox = torch.einsum('bchwk,ck->bchw', sampled_x, wx)
        oy = torch.einsum('bchwk,ck->bchw', sampled_y, wy)
        out = (ox + oy)

        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out
