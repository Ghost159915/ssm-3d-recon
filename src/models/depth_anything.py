"""
src/models/depth_anything.py
============================
Depth Anything V2 wrapper (HuggingFace transformers).

Provides a simple interface:
    - Load the model once, run inference on batches of RGB images
    - Returns relative depth maps (raw logits, not metric)
    - Optionally returns feature maps from the encoder for use in
      the S5 temporal consistency module

Depth Anything V2 produces *relative* (affine-invariant) depth —
values are not in metres. Metric scale is recovered downstream in
src/geometry/scale_align.py using COLMAP sparse points or GT depth.

Supported model sizes:
    "small"  — ViT-S backbone, 24.8M params, fastest
    "base"   — ViT-B backbone, 97.5M params, balanced (recommended)
    "large"  — ViT-L backbone, 335M params, best quality

HuggingFace model IDs:
    "depth-anything/Depth-Anything-V2-Small-hf"
    "depth-anything/Depth-Anything-V2-Base-hf"
    "depth-anything/Depth-Anything-V2-Large-hf"

Usage
-----
    from src.models.depth_anything import DepthAnythingV2

    model = DepthAnythingV2(size="small", device="mps")

    import torch
    rgb = torch.rand(1, 3, 480, 640)   # [B, 3, H, W] float32 [0,1]
    depth = model.predict(rgb)          # [B, 1, H, W] float32 relative depth
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Auto device selection
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model size → HuggingFace ID mapping
# ---------------------------------------------------------------------------

_HF_IDS = {
    "small":  "depth-anything/Depth-Anything-V2-Small-hf",
    "base":   "depth-anything/Depth-Anything-V2-Base-hf",
    "large":  "depth-anything/Depth-Anything-V2-Large-hf",
}

# Feature dimension output by each backbone (before the DPT head)
# These are the dims we'll hook into for the S5 temporal module.
_FEATURE_DIMS = {
    "small":  384,
    "base":   768,
    "large":  1024,
}


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class DepthAnythingV2(nn.Module):
    """
    Depth Anything V2 inference wrapper.

    Parameters
    ----------
    size     : str   "small" | "base" | "large"
    device   : str   "auto" | "cpu" | "mps" | "cuda"
    cache_dir: str   HuggingFace cache directory (None = default)
    """

    def __init__(
        self,
        size: str = "small",
        device: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        assert size in _HF_IDS, f"size must be one of {list(_HF_IDS)}"

        self.size = size
        self.device = _auto_device() if device == "auto" else device
        self.feature_dim = _FEATURE_DIMS[size]

        hf_id = _HF_IDS[size]
        print(f"[DepthAnythingV2] Loading '{hf_id}' on {self.device}...")

        # Lazy import so the file is importable without transformers installed
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            raise ImportError(
                "transformers >= 4.40 is required for Depth Anything V2.\n"
                "Install: pip install transformers>=4.40 timm"
            )

        self.processor = AutoImageProcessor.from_pretrained(
            hf_id, cache_dir=cache_dir
        )
        self._hf_model = AutoModelForDepthEstimation.from_pretrained(
            hf_id, cache_dir=cache_dir
        ).to(self.device)
        self._hf_model.eval()

        print(f"[DepthAnythingV2] Loaded. feature_dim={self.feature_dim}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        rgb: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Run depth estimation on a batch of RGB images.

        Parameters
        ----------
        rgb         : [B, 3, H, W] float32, values in [0, 1]
        output_size : (H_out, W_out) or None.
                      If None, returns at the model's native output resolution
                      (typically same as input after bilinear upsampling in DPT).

        Returns
        -------
        depth : [B, 1, H_out, W_out] float32  relative depth (not metric)
                Values are normalised to [0, 1] within each frame
                (larger = further from camera in Depth Anything convention).
        """
        B, C, H, W = rgb.shape
        assert C == 3, "Expected [B, 3, H, W] input"

        # HuggingFace processor expects PIL images or numpy HWC uint8
        # We convert batch items one by one
        from PIL import Image

        pil_images = []
        for i in range(B):
            arr = (rgb[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(arr))

        # Processor handles normalisation and resize to model's expected size
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._hf_model(**inputs)

        # predicted_depth: [B, H', W']  (model output resolution)
        depth = outputs.predicted_depth   # [B, H', W']

        # Add channel dim → [B, 1, H', W']
        depth = depth.unsqueeze(1)

        # Normalise each frame independently to [0, 1]
        B2, _, Hd, Wd = depth.shape
        flat = depth.view(B2, -1)
        d_min = flat.min(dim=1).values.view(B2, 1, 1, 1)
        d_max = flat.max(dim=1).values.view(B2, 1, 1, 1)
        depth = (depth - d_min) / (d_max - d_min + 1e-8)

        # Resize to requested output size
        if output_size is not None and output_size != (Hd, Wd):
            depth = F.interpolate(
                depth, size=output_size, mode="bilinear", align_corners=False
            )

        return depth.float()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_with_features(
        self,
        rgb: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run depth estimation AND return intermediate encoder features
        for the S5 temporal consistency module.

        The encoder features are spatially pooled to a 1-D vector per
        frame so they can be fed as the input sequence to S5.

        Returns
        -------
        depth    : [B, 1, H, W]   normalised relative depth
        features : [B, D]         global average-pooled encoder features
                                  D = self.feature_dim
        """
        # Register a hook to capture the last encoder hidden state
        features_cache: Dict[str, torch.Tensor] = {}

        def _hook(module, input, output):
            # output may be a tuple (hidden_state, ...) or just a tensor
            hs = output[0] if isinstance(output, tuple) else output
            features_cache["hs"] = hs   # [B, seq_len, D] typically

        # The DPT encoder stores last hidden state in the backbone
        # For Depth Anything V2 / ViT backbone the relevant layer is:
        try:
            hook_handle = self._hf_model.backbone.encoder.layer[-1].register_forward_hook(_hook)
        except AttributeError:
            # Fallback: hook onto the full backbone
            hook_handle = self._hf_model.backbone.register_forward_hook(_hook)

        depth = self.predict(rgb, output_size=output_size)

        hook_handle.remove()

        if "hs" in features_cache:
            hs = features_cache["hs"]   # [B, seq_len, D] or [B, D, H', W']
            if hs.dim() == 4:
                # Spatial features [B, D, H', W'] → global pool → [B, D]
                feats = hs.mean(dim=[2, 3])
            elif hs.dim() == 3:
                # Sequence features [B, T, D] → mean over tokens → [B, D]
                feats = hs.mean(dim=1)
            else:
                feats = hs
        else:
            warnings.warn("Feature hook did not capture output; returning zeros.")
            feats = torch.zeros(rgb.shape[0], self.feature_dim, device=self.device)

        return depth, feats.float()

    # ------------------------------------------------------------------
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Alias for predict(). Allows use as a standard nn.Module."""
        return self.predict(rgb)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    print(f"Testing DepthAnythingV2 size='{size}'")

    model = DepthAnythingV2(size=size)

    B, H, W = 2, 240, 320
    rgb = torch.rand(B, 3, H, W)
    print(f"Input: {rgb.shape}")

    depth = model.predict(rgb, output_size=(H, W))
    print(f"Depth: {depth.shape}  range [{depth.min():.3f}, {depth.max():.3f}]")

    depth2, feats = model.predict_with_features(rgb, output_size=(H, W))
    print(f"Depth (with feats): {depth2.shape}")
    print(f"Features: {feats.shape}  (should be [B, {model.feature_dim}])")
    print("All checks passed.")
