"""
src/utils/metrics.py
====================
Quantitative depth evaluation metrics.

Metrics used in the TUM RGB-D benchmark and monocular depth literature:
    - AbsRel  : mean absolute relative error            (lower is better)
    - RMSE    : root mean squared error                 (lower is better)
    - SqRel   : mean squared relative error             (lower is better)
    - δ < 1.25 : % pixels with max(d/d*, d*/d) < 1.25  (higher is better)
    - δ < 1.25²
    - δ < 1.25³

Usage
-----
    from src.utils.metrics import compute_depth_metrics, MetricsAccumulator

    # Single frame
    metrics = compute_depth_metrics(d_pred, d_gt)

    # Full dataset
    acc = MetricsAccumulator()
    for d_pred, d_gt in dataloader:
        acc.update(d_pred, d_gt)
    results = acc.summary()
    print(results)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def compute_depth_metrics(
    d_pred: torch.Tensor,
    d_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_depth: float = 10.0,
) -> Dict[str, float]:
    """
    Compute standard depth evaluation metrics.

    Parameters
    ----------
    d_pred   : [H, W] or [1, H, W] or [B, H, W]  predicted metric depth
    d_gt     : same shape as d_pred                GT metric depth (metres)
    mask     : bool tensor, same shape. None → derived from d_gt > 0
    max_depth: clip both depths to this value (metres)

    Returns
    -------
    Dict with keys: abs_rel, rmse, sq_rel, delta_1, delta_2, delta_3
    All values are Python floats.
    """
    # Flatten to 1-D
    d_pred = d_pred.float().squeeze().cpu()
    d_gt   = d_gt.float().squeeze().cpu()

    if mask is None:
        mask = (d_gt > 0) & (d_gt < max_depth)

    d_pred = d_pred[mask]
    d_gt   = d_gt[mask]

    # Clamp predictions
    d_pred = d_pred.clamp(1e-4, max_depth)

    # --- AbsRel ---
    abs_rel = ((d_pred - d_gt).abs() / d_gt).mean().item()

    # --- SqRel ---
    sq_rel = (((d_pred - d_gt) ** 2) / d_gt).mean().item()

    # --- RMSE ---
    rmse = torch.sqrt(((d_pred - d_gt) ** 2).mean()).item()

    # --- δ thresholds ---
    ratio = torch.max(d_pred / d_gt, d_gt / d_pred)
    delta_1 = (ratio < 1.25).float().mean().item()
    delta_2 = (ratio < 1.25 ** 2).float().mean().item()
    delta_3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "abs_rel": abs_rel,
        "sq_rel":  sq_rel,
        "rmse":    rmse,
        "delta_1": delta_1,   # δ < 1.25
        "delta_2": delta_2,   # δ < 1.25²
        "delta_3": delta_3,   # δ < 1.25³
    }


# ---------------------------------------------------------------------------
# 3D mesh metrics (F-score, Completeness, Accuracy)
# ---------------------------------------------------------------------------

def compute_mesh_fscore(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Compute F-score between a predicted point cloud and a GT point cloud
    at a given distance threshold.

    Parameters
    ----------
    pred_pts  : [N, 3]  predicted surface samples
    gt_pts    : [M, 3]  GT surface samples
    threshold : float   distance threshold in metres (5 cm default)

    Returns
    -------
    Dict with keys: precision, recall, fscore, completeness, accuracy
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("scipy is required for mesh F-score: pip install scipy")

    # Precision: how much of pred is within threshold of GT
    tree_gt = cKDTree(gt_pts)
    dist_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    precision = (dist_pred_to_gt < threshold).mean()

    # Recall / Completeness: how much of GT is within threshold of pred
    tree_pred = cKDTree(pred_pts)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)
    recall = (dist_gt_to_pred < threshold).mean()

    fscore = (
        2 * precision * recall / (precision + recall + 1e-8)
    )

    return {
        "precision":    float(precision),   # = accuracy
        "recall":       float(recall),      # = completeness
        "fscore":       float(fscore),
        "accuracy":     float(precision),
        "completeness": float(recall),
    }


# ---------------------------------------------------------------------------
# Accumulator: average metrics over many frames
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """
    Running accumulator for depth metrics over a dataset.

    Usage
    -----
        acc = MetricsAccumulator()
        for batch in dataloader:
            acc.update(d_pred, d_gt)
        print(acc.summary())
    """

    def __init__(self):
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def update(
        self,
        d_pred: torch.Tensor,
        d_gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Update accumulator with a new frame (or batch of frames)."""
        # Handle batched input [B, H, W]
        if d_pred.dim() == 3 and d_pred.shape[0] > 1:
            for i in range(d_pred.shape[0]):
                m = mask[i] if mask is not None else None
                self.update(d_pred[i], d_gt[i], m)
            return

        metrics = compute_depth_metrics(d_pred, d_gt, mask)
        for k, v in metrics.items():
            self._totals[k] = self._totals.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def summary(self) -> Dict[str, float]:
        """Return mean metrics over all updated frames."""
        return {k: self._totals[k] / self._counts[k] for k in self._totals}

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()

    def pretty_print(self, label: str = "Metrics") -> None:
        results = self.summary()
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
        print(f"  AbsRel  : {results['abs_rel']:.4f}  (lower ↓)")
        print(f"  SqRel   : {results['sq_rel']:.4f}  (lower ↓)")
        print(f"  RMSE    : {results['rmse']:.4f} m (lower ↓)")
        print(f"  δ<1.25  : {results['delta_1']*100:.2f}%  (higher ↑)")
        print(f"  δ<1.25² : {results['delta_2']*100:.2f}%  (higher ↑)")
        print(f"  δ<1.25³ : {results['delta_3']*100:.2f}%  (higher ↑)")
        print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    H, W = 64, 80

    # Perfect prediction
    d_gt   = torch.rand(H, W) * 4 + 0.5
    d_pred = d_gt.clone()
    m = compute_depth_metrics(d_pred, d_gt)
    print("Perfect prediction:")
    print(f"  abs_rel={m['abs_rel']:.6f} (expect ~0)")
    print(f"  delta_1={m['delta_1']:.6f} (expect 1.0)")

    # Noisy prediction
    d_pred_noisy = d_gt + torch.randn(H, W) * 0.2
    m2 = compute_depth_metrics(d_pred_noisy, d_gt)
    print(f"\nNoisy prediction:")
    print(f"  abs_rel={m2['abs_rel']:.4f}")
    print(f"  rmse   ={m2['rmse']:.4f}")

    # Accumulator test
    acc = MetricsAccumulator()
    for _ in range(5):
        acc.update(d_pred_noisy, d_gt)
    acc.pretty_print("Accumulated (5 frames)")
