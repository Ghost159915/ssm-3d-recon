"""
src/models/temporal_depth.py
============================
TemporalDepthConsistency — the core SSM-powered contribution.

Architecture
------------
This module replaces the GRU in NeuralRecon's cross-fragment fusion
(CVPR 2021, Table 5) with an S5 SSM — the same architectural
substitution made in the thesis (S5 replacing ConvLSTM in S5-RVT).

Pipeline:
    RGB frames [T, 3, H, W]
        │
        ▼
    CNN Encoder (lightweight, per-frame)  ──→  features [T, D, H', W']
        │
    spatial flatten + linear project
        │
        ▼
    [T, H'*W', D_model]  (sequence = positions across space, for each frame)
        │
    S5 temporal axis: run S5 over the T (time) dimension
    For each spatial position independently:
        ▼
    [T, H'*W', D_model]
        │
    Depth Head (per-pixel MLP)
        │
        ▼
    Refined depth maps [T, 1, H, W]

Why this works
--------------
The S5 hidden state h_t accumulates a "scene memory" across frames —
exactly what NeuralRecon's GRU hidden state H_t^g does, but with:
  - Better long-range memory (S4/S5 vs GRU gradient flow)
  - Variable step sizes Δ (can represent irregular frame timing)
  - Parallel training (associative scan vs sequential backprop)
  - Linear complexity in sequence length (vs quadratic for attention)

Usage
-----
    from src.models.temporal_depth import TemporalDepthConsistency

    model = TemporalDepthConsistency(
        img_size=(240, 320),
        d_model=128,
        d_state=64,
        n_layers=3,
    )

    rgb = torch.rand(8, 3, 240, 320)    # 8-frame sequence
    refined = model(rgb)                # [8, 1, 240, 320]
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from .s5 import S5Stack
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from s5 import S5Stack


# ---------------------------------------------------------------------------
# CNN Encoder (lightweight — feature extractor, not full depth network)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """
    Lightweight CNN encoder: extracts spatial depth-relevant features
    from a single RGB frame.

    Input : [B, 3, H, W]
    Output: [B, C_out, H//8, W//8]   (stride-8 downsampling)

    Architecture: 3 conv blocks (conv → BN → ReLU), doubling channels each time.
    Kept small intentionally — the S5 provides the temporal depth; the CNN
    just extracts local texture/edge features for each frame independently.
    """

    def __init__(self, c_out: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 3 → 32, stride 2  (H/2)
            nn.Conv2d(3,  32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 32 → 64, stride 2  (H/4)
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 64 → c_out, stride 2  (H/8)
            nn.Conv2d(64, c_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.c_out = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)   # [B, c_out, H//8, W//8]


class CNNEncoderFlex(nn.Module):
    """Same as CNNEncoder but accepts variable input channels (c_in)."""

    def __init__(self, c_in: int = 4, c_out: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(c_in, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, c_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.c_out = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Depth Head (per-pixel, from SSM features → depth)
# ---------------------------------------------------------------------------

class DepthHead(nn.Module):
    """
    Predict a depth value from the temporally-fused feature vector.

    Input : [B, D_model]  per spatial position
    Output: [B, 1]        depth (relative, in [0,1])

    Note: The head is applied per-pixel after the S5 output is reshaped back
    to spatial maps, then bilinearly upsampled to the original resolution.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),   # output in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)   # [B, 1] or [..., 1]


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class TemporalDepthConsistency(nn.Module):
    """
    SSM-based temporal depth consistency module.

    This is the architectural contribution of the project:
    an S5 SSM replaces GRU in the cross-frame temporal fusion slot.

    Parameters
    ----------
    img_size     : (H, W)   input image resolution
    cnn_channels : int      CNN encoder output channels (= D_spatial)
    d_model      : int      S5 model dimension
    d_state      : int      S5 hidden state size (per layer)
    n_layers     : int      number of S5 blocks
    dropout      : float    dropout inside S5 FFN
    use_parallel : bool     True = parallel scan (training), False = recurrent
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (240, 320),
        cnn_channels: int = 64,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_parallel: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.use_parallel = use_parallel

        H, W = img_size
        self.feat_h = H // 8
        self.feat_w = W // 8

        # CNN: extract per-frame features
        self.encoder = CNNEncoder(c_out=cnn_channels)

        # Project CNN features to S5 model dimension
        self.input_proj = nn.Linear(cnn_channels, d_model)

        # S5 stack: temporal fusion across frames
        # We process each spatial position as an independent sequence over time.
        # The "sequence" is T frames; the "batch" is B * H' * W' positions.
        self.ssm = S5Stack(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Depth head: fused features → relative depth
        self.depth_head = DepthHead(d_model)

    # ------------------------------------------------------------------
    def forward(
        self,
        rgb_seq: torch.Tensor,
        use_parallel: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        rgb_seq     : [T, 3, H, W] or [B, T, 3, H, W]
                      A sequence of T RGB frames. If 4-D, treated as single
                      batch (B=1 implicitly); if 5-D, B is the batch dim.
        use_parallel: override instance default

        Returns
        -------
        depth_refined : [T, 1, H, W] or [B, T, 1, H, W]  relative depth
                        Upsampled to original (H, W) via bilinear interpolation.
        """
        scan_parallel = use_parallel if use_parallel is not None else self.use_parallel

        # --- Handle 4-D input (T, 3, H, W) → add batch dim ---
        squeeze = False
        if rgb_seq.dim() == 4:
            rgb_seq = rgb_seq.unsqueeze(0)   # [1, T, 3, H, W]
            squeeze = True

        B, T, C, H, W = rgb_seq.shape

        # --- CNN encoder: run independently on each (B, T) frame ---
        # Merge B and T into a single batch dim for the CNN
        rgb_flat = rearrange(rgb_seq, "b t c h w -> (b t) c h w")
        feat_flat = self.encoder(rgb_flat)    # [(B*T), cnn_ch, H', W']
        _, Cf, Hf, Wf = feat_flat.shape

        # --- Project to d_model ---
        # Reshape spatial → sequence dim: [(B*T), H'*W', cnn_ch]
        feat_seq = rearrange(feat_flat, "(bt) c h w -> bt (h w) c", bt=B * T)
        feat_seq = self.input_proj(feat_seq)  # [(B*T), H'*W', d_model]

        # --- S5 temporal fusion ---
        # We want S5 to see the T frames as a sequence.
        # Current shape: [(B*T), H'*W', D]
        # We need: [B * H'*W', T, D]  (each spatial position as a sequence over time)
        # Reshape: (B*T) = B×T, spatial = H'×W'
        feat_seq = rearrange(feat_seq, "(b t) s d -> (b s) t d", b=B, t=T)
        # feat_seq: [B*H'*W', T, D]

        # Run S5 over the time dimension T
        fused = self.ssm(feat_seq, use_parallel=scan_parallel)  # [B*H'*W', T, D]

        # --- Depth head: predict depth at each (spatial, time) position ---
        # fused: [B*H'*W', T, D]
        depth_pred = self.depth_head(fused)     # [B*H'*W', T, 1]

        # --- Reshape back to spatial maps ---
        # [B*H'*W', T, 1] → [B, T, 1, H', W']
        depth_maps = rearrange(
            depth_pred, "(b h w) t 1 -> b t 1 h w",
            b=B, h=Hf, w=Wf
        )

        # --- Upsample to original resolution ---
        depth_maps = rearrange(depth_maps, "b t c h w -> (b t) c h w")
        depth_up = F.interpolate(
            depth_maps, size=(H, W), mode="bilinear", align_corners=False
        )
        depth_up = rearrange(depth_up, "(b t) 1 h w -> b t 1 h w", b=B, t=T)

        if squeeze:
            depth_up = depth_up.squeeze(0)   # [T, 1, H, W]

        return depth_up


# ---------------------------------------------------------------------------
# Edge-aware smoothness loss (used in training script)
# ---------------------------------------------------------------------------

def edge_aware_smoothness_loss(
    depth: torch.Tensor,
    rgb: torch.Tensor,
) -> torch.Tensor:
    """
    Penalise depth gradients where image gradients are small.
    Encourages smooth depth in textureless regions and preserves
    depth discontinuities at edges.

    Parameters
    ----------
    depth : [B, 1, H, W]   predicted depth
    rgb   : [B, 3, H, W]   corresponding RGB (for edge weights)

    Returns
    -------
    loss : scalar tensor
    """
    # Depth gradients
    d_dx = depth[:, :, :, :-1] - depth[:, :, :, 1:]    # horizontal
    d_dy = depth[:, :, :-1, :] - depth[:, :, 1:, :]    # vertical

    # Image gradients (convert to grayscale for edge detection)
    gray = rgb.mean(dim=1, keepdim=True)
    i_dx = (gray[:, :, :, :-1] - gray[:, :, :, 1:]).abs()
    i_dy = (gray[:, :, :-1, :] - gray[:, :, 1:, :]).abs()

    # Edge-aware weighting: suppress depth smoothing where edges exist
    w_dx = torch.exp(-i_dx)
    w_dy = torch.exp(-i_dy)

    loss = (w_dx * d_dx.abs()).mean() + (w_dy * d_dy.abs()).mean()
    return loss


# ---------------------------------------------------------------------------
# DepthRefinementSSM — takes DAV2 depth hints + RGB, outputs refined depth
# ---------------------------------------------------------------------------

class DepthRefinementSSM(nn.Module):
    """
    S5-based temporal depth refinement.

    Takes pre-computed per-frame depth estimates (from Depth Anything V2)
    as input alongside RGB, and uses an S5 SSM to learn temporally
    consistent corrections.

    This is the architecturally correct approach:
    - DAV2 handles the hard single-frame depth estimation (it's good at that)
    - S5 handles the temporal fusion / consistency (that's its strength)
    - The learning task is: "given DAV2's guess + neighbours, improve it"
    - This mirrors NeuralRecon's GRU slot: pre-computed features in, refined state out

    Input:
        depth_hint : [T, 1, H, W]  scale-aligned depth from DAV2 (normalised [0,1])
        rgb        : [T, 3, H, W]  corresponding RGB frames

    Output:
        depth_refined : [T, 1, H, W]  temporally consistent depth (normalised [0,1])
                        = depth_hint + small learned residual correction
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (240, 320),
        cnn_channels: int = 32,
        d_model: int = 64,
        d_state: int = 32,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_parallel: bool = True,
    ):
        super().__init__()
        self.img_size   = img_size
        self.use_parallel = use_parallel

        H, W = img_size
        self.feat_h = H // 8
        self.feat_w = W // 8

        # 4-channel input: depth hint (1) + RGB (3)
        self.encoder    = CNNEncoderFlex(c_in=4, c_out=cnn_channels)
        self.input_proj = nn.Linear(cnn_channels, d_model)

        self.ssm = S5Stack(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Residual head: output a small additive correction in (-0.5, 0.5)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),   # bounded correction
        )

    def forward(
        self,
        depth_hint: torch.Tensor,
        rgb: torch.Tensor,
        use_parallel: Optional[bool] = None,
    ) -> torch.Tensor:
        scan_parallel = use_parallel if use_parallel is not None else self.use_parallel

        squeeze = False
        if depth_hint.dim() == 4 and depth_hint.shape[0] != rgb.shape[0]:
            # batch dim mismatch — shouldn't happen
            pass
        # Handle [T, 1, H, W] input (no explicit batch dim) by adding B=1
        if depth_hint.dim() == 4 and rgb.dim() == 4:
            # Already [T/B, C, H, W] — treat T as batch through CNN, then reshape
            T = depth_hint.shape[0]
            B = 1
            squeeze = True
        else:
            B, T = depth_hint.shape[:2]

        H, W = depth_hint.shape[-2], depth_hint.shape[-1]

        # --- Concatenate depth hint + RGB → 4-channel input ---
        if squeeze:
            # depth_hint: [T, 1, H, W], rgb: [T, 3, H, W]
            x4 = torch.cat([depth_hint, rgb], dim=1)   # [T, 4, H, W]
        else:
            # [B, T, 1, H, W] and [B, T, 3, H, W]
            x4 = torch.cat([depth_hint, rgb], dim=2)   # [B, T, 4, H, W]
            x4 = rearrange(x4, "b t c h w -> (b t) c h w")
            T_actual = T

        if squeeze:
            x4_flat = x4   # [T, 4, H, W]
        else:
            x4_flat = x4   # [(B*T), 4, H, W]

        # --- CNN encode ---
        feat_flat = self.encoder(x4_flat)   # [(B*T or T), cnn_ch, H', W']
        _, Cf, Hf, Wf = feat_flat.shape

        # Reshape → sequence for S5: [B*Hf*Wf, T, d_model]
        if squeeze:
            feat_seq = rearrange(feat_flat, "t c h w -> (h w) t c")   # [H'W', T, C]
        else:
            feat_seq = rearrange(feat_flat, "(b t) c h w -> (b h w) t c", b=B, t=T)

        feat_seq = self.input_proj(feat_seq)   # [B*H'W', T, d_model]

        # --- S5 temporal fusion ---
        fused = self.ssm(feat_seq, use_parallel=scan_parallel)  # [B*H'W', T, d_model]

        # --- Residual correction ---
        delta = self.residual_head(fused) * 0.2   # [B*H'W', T, 1]  small correction

        # --- Reshape back to spatial ---
        if squeeze:
            delta_map = rearrange(delta, "(h w) t 1 -> t 1 h w", h=Hf, w=Wf)
        else:
            delta_map = rearrange(delta, "(b h w) t 1 -> b t 1 h w", b=B, h=Hf, w=Wf)

        # Upsample delta to original resolution
        if squeeze:
            delta_up = F.interpolate(delta_map, size=(H, W), mode="bilinear", align_corners=False)
        else:
            delta_flat = rearrange(delta_map, "b t c h w -> (b t) c h w")
            delta_up = F.interpolate(delta_flat, size=(H, W), mode="bilinear", align_corners=False)
            delta_up = rearrange(delta_up, "(b t) 1 h w -> b t 1 h w", b=B, t=T)

        # Residual: add correction to the depth hint
        refined = depth_hint + delta_up
        refined = refined.clamp(0.0, 1.0)
        return refined


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    torch.manual_seed(0)

    T, H, W = 10, 240, 320
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = TemporalDepthConsistency(
        img_size=(H, W),
        cnn_channels=32,
        d_model=64,
        d_state=32,
        n_layers=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    rgb_seq = torch.rand(T, 3, H, W).to(device)

    # Parallel mode (training)
    t0 = time.time()
    depth_out = model(rgb_seq, use_parallel=True)
    print(f"Output [parallel]: {depth_out.shape}  dt={time.time()-t0:.2f}s")

    # Recurrent mode (inference)
    t0 = time.time()
    depth_out2 = model(rgb_seq, use_parallel=False)
    print(f"Output [recurrent]: {depth_out2.shape}  dt={time.time()-t0:.2f}s")

    # Loss check
    dummy_gt = torch.rand_like(depth_out)
    l1_loss = F.l1_loss(depth_out, dummy_gt)
    smooth_loss = edge_aware_smoothness_loss(depth_out, rgb_seq)
    print(f"L1 loss: {l1_loss:.4f}")
    print(f"Smoothness loss: {smooth_loss:.4f}")
    print("All checks passed.")
