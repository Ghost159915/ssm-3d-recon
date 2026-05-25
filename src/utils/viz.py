"""
src/utils/viz.py
================
Visualisation helpers for depth maps, error plots, and 3D scenes.

Usage
-----
    from src.utils.viz import (
        plot_depth_comparison,
        colorize_depth,
        save_depth_grid,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Depth map colourisation
# ---------------------------------------------------------------------------

def colorize_depth(
    depth: Union[np.ndarray, torch.Tensor],
    cmap: str = "plasma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    invalid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert a float depth map to an RGB image using a matplotlib colormap.

    Parameters
    ----------
    depth        : [H, W] float32   depth values (metres or relative)
    cmap         : str               matplotlib colormap name
    vmin, vmax   : float | None      clipping range (None = auto)
    invalid_mask : bool [H, W]       pixels to show as black

    Returns
    -------
    rgb : [H, W, 3] uint8
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().float().numpy()
    depth = depth.squeeze().astype(np.float32)

    # Auto range from valid pixels
    valid = (depth > 0) if invalid_mask is None else ~invalid_mask
    if vmin is None:
        vmin = float(depth[valid].min()) if valid.any() else 0.0
    if vmax is None:
        vmax = float(depth[valid].max()) if valid.any() else 1.0

    normed = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cm = plt.get_cmap(cmap)
    rgb = (cm(normed)[:, :, :3] * 255).astype(np.uint8)

    if invalid_mask is not None:
        rgb[invalid_mask] = 0
    elif (depth == 0).any():
        rgb[depth == 0] = 0

    return rgb


# ---------------------------------------------------------------------------
# Side-by-side comparison plot
# ---------------------------------------------------------------------------

def plot_depth_comparison(
    rgb: Union[np.ndarray, torch.Tensor],
    depth_gt: Union[np.ndarray, torch.Tensor],
    depth_pred: Union[np.ndarray, torch.Tensor],
    depth_refined: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: str = "Depth Comparison",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot RGB | GT depth | predicted depth | (optional) refined depth | error map.

    Parameters
    ----------
    rgb            : [H, W, 3] uint8 or [3, H, W] float32
    depth_gt       : [H, W]    metres
    depth_pred     : [H, W]    metres (scale-aligned)
    depth_refined  : [H, W]    metres (SSM-refined, optional)
    title          : str
    save_path      : str | None   save figure to this path
    show           : bool         call plt.show()
    """
    # Convert tensors
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    rgb = _to_np(rgb)
    if rgb.shape[0] == 3:                        # [3, H, W] → [H, W, 3]
        rgb = rgb.transpose(1, 2, 0)
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    d_gt  = _to_np(depth_gt).squeeze().astype(np.float32)
    d_pre = _to_np(depth_pred).squeeze().astype(np.float32)

    valid = d_gt > 0
    vmin = float(d_gt[valid].min()) if valid.any() else 0.0
    vmax = float(d_gt[valid].max()) if valid.any() else 5.0

    n_panels = 4 if depth_refined is None else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    fig.suptitle(title, fontsize=12)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(colorize_depth(d_gt, vmin=vmin, vmax=vmax))
    axes[1].set_title("GT Depth")
    axes[1].axis("off")

    axes[2].imshow(colorize_depth(d_pre, vmin=vmin, vmax=vmax))
    axes[2].set_title("Predicted Depth")
    axes[2].axis("off")

    if depth_refined is not None:
        d_ref = _to_np(depth_refined).squeeze().astype(np.float32)
        axes[3].imshow(colorize_depth(d_ref, vmin=vmin, vmax=vmax))
        axes[3].set_title("SSM Refined")
        axes[3].axis("off")
        err = np.abs(d_ref - d_gt) * valid
        err_ax = axes[4]
    else:
        err = np.abs(d_pre - d_gt) * valid
        err_ax = axes[3]

    err_ax.imshow(colorize_depth(err, cmap="hot", vmin=0))
    err_ax.set_title("Abs Error (m)")
    err_ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[viz] Saved → {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Save a grid of depth frames
# ---------------------------------------------------------------------------

def save_depth_grid(
    depths: Union[List, torch.Tensor, np.ndarray],
    save_path: str,
    n_cols: int = 4,
    title: str = "",
    cmap: str = "plasma",
) -> None:
    """
    Save a grid of depth maps as a PNG.

    Parameters
    ----------
    depths    : list of [H, W] arrays, or [N, H, W] tensor
    save_path : output PNG path
    n_cols    : number of columns in the grid
    """
    if isinstance(depths, torch.Tensor):
        depths = [depths[i].cpu().numpy() for i in range(len(depths))]
    elif isinstance(depths, np.ndarray) and depths.ndim == 3:
        depths = [depths[i] for i in range(len(depths))]

    N = len(depths)
    n_rows = (N + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if title:
        fig.suptitle(title)
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i, d in enumerate(depths):
        r, c = divmod(i, n_cols)
        axes[r, c].imshow(colorize_depth(d, cmap=cmap))
        axes[r, c].set_title(f"Frame {i}")
        axes[r, c].axis("off")

    # Hide unused panels
    for i in range(N, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved depth grid → {save_path}")


# ---------------------------------------------------------------------------
# Print metrics table
# ---------------------------------------------------------------------------

def print_metrics_table(
    results: Dict[str, Dict[str, float]],
    title: str = "Depth Evaluation Results",
) -> None:
    """
    Print a formatted table of metrics.

    Parameters
    ----------
    results : {"Method Name": {"abs_rel": ..., "rmse": ..., ...}, ...}
    """
    methods = list(results.keys())
    keys    = ["abs_rel", "sq_rel", "rmse", "delta_1", "delta_2", "delta_3"]
    headers = ["AbsRel↓", "SqRel↓", "RMSE↓", "δ<1.25↑", "δ<1.25²↑", "δ<1.25³↑"]

    col_w = 12
    name_w = max(len(m) for m in methods) + 2

    sep = "-" * (name_w + col_w * len(keys))
    print(f"\n{title}")
    print(sep)
    header_row = f"{'Method':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in headers)
    print(header_row)
    print(sep)

    for method, m in results.items():
        row = f"{method:<{name_w}}"
        for k in keys:
            v = m.get(k, float("nan"))
            if "delta" in k:
                row += f"{v*100:>{col_w}.2f}"
            else:
                row += f"{v:>{col_w}.4f}"
        print(row)
    print(sep + "\n")
