"""
src/geometry/scale_align.py
===========================
Metric scale alignment for monocular depth estimates.

Depth Anything V2 produces *relative* (affine-invariant) depth — the
values are not in metres. To integrate depths into a metric TSDF volume
we need to recover the unknown scale s and shift t such that:

    d_metric ≈ s * d_mono + t

where d_metric comes from sparse COLMAP points or, in our case, the TUM
ground-truth depth maps.

Method: least-squares linear regression over valid pixel pairs.

Usage
-----
    from src.geometry.scale_align import align_depth, align_depth_to_gt

    # Align predicted depth to GT (used for evaluation)
    aligned = align_depth_to_gt(d_pred, d_gt, mask=d_gt > 0)

    # Align to COLMAP sparse points (used in real-world pipeline)
    aligned = align_depth_to_colmap(d_pred, sparse_xyz, K, pose)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Core: least-squares scale + shift
# ---------------------------------------------------------------------------

def least_squares_scale_shift(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Find (s, t) minimising  ||s * pred + t - gt||²  over valid pixels.

    Parameters
    ----------
    pred : array [N]   predicted depth (relative)
    gt   : array [N]   ground-truth depth (metric, metres)
    mask : bool array [N]  True = use this pixel. None = use all.

    Returns
    -------
    s : float   scale
    t : float   shift
    """
    if mask is not None:
        pred = pred[mask]
        gt   = gt[mask]

    if len(pred) < 10:
        # Not enough valid points — return identity
        return 1.0, 0.0

    # Linear system: [pred | 1] @ [s, t]^T = gt
    A = np.stack([pred, np.ones_like(pred)], axis=1)   # [N, 2]
    # Least squares: (A^T A)^{-1} A^T gt
    result = np.linalg.lstsq(A, gt, rcond=None)
    s, t = result[0]
    return float(s), float(t)


# ---------------------------------------------------------------------------
# Align predicted depth to GT depth map (evaluation / TUM pipeline)
# ---------------------------------------------------------------------------

def align_depth_to_gt(
    d_pred: torch.Tensor,
    d_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Align relative depth prediction to metric GT depth using least-squares.

    Computes s, t on the valid GT pixels, then applies:
        d_aligned = s * d_pred + t

    Parameters
    ----------
    d_pred : [H, W] or [1, H, W]  float32  relative depth
    d_gt   : [H, W] or [1, H, W]  float32  metric depth (metres)
    mask   : [H, W] bool   valid pixels (d_gt > 0 and within sensor range)
             If None, mask is derived from d_gt > 0.

    Returns
    -------
    d_aligned : same shape as d_pred, float32, in metres
    """
    # Squeeze to [H, W]
    if d_pred.dim() == 3:
        d_pred = d_pred.squeeze(0)
    if d_gt.dim() == 3:
        d_gt = d_gt.squeeze(0)

    if mask is None:
        mask = d_gt > 0

    # To numpy for lstsq
    pred_np = d_pred.cpu().float().numpy()
    gt_np   = d_gt.cpu().float().numpy()
    mask_np = mask.cpu().bool().numpy()

    s, t = least_squares_scale_shift(pred_np.ravel(), gt_np.ravel(), mask_np.ravel())

    d_aligned = s * d_pred + t
    # Clamp: depth cannot be negative
    d_aligned = d_aligned.clamp(min=0.0)

    return d_aligned


# ---------------------------------------------------------------------------
# Align to COLMAP sparse 3D points
# ---------------------------------------------------------------------------

def align_depth_to_colmap(
    d_pred: torch.Tensor,
    sparse_xyz: np.ndarray,
    K: np.ndarray,
    pose_c2w: np.ndarray,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """
    Align relative depth to metric scale using COLMAP sparse 3D points.

    Projects COLMAP 3D points into the image frame, reads off the predicted
    depth at those pixel locations, and fits a least-squares scale+shift.

    Parameters
    ----------
    d_pred     : [H, W]   predicted relative depth
    sparse_xyz : [M, 3]   COLMAP 3D point coordinates (world frame)
    K          : [3, 3]   camera intrinsic matrix
    pose_c2w   : [4, 4]   camera-to-world SE(3) (from COLMAP / TUM GT)
    img_h, img_w : int    image dimensions

    Returns
    -------
    d_aligned : [H, W] float32  metric depth
    """
    # World → camera: invert pose_c2w
    pose_w2c = np.linalg.inv(pose_c2w)   # [4, 4]
    R = pose_w2c[:3, :3]
    t_vec = pose_w2c[:3, 3]

    # Transform 3D points to camera frame
    pts_cam = (R @ sparse_xyz.T).T + t_vec   # [M, 3]

    # Only keep points in front of the camera
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]

    if len(pts_cam) < 10:
        # Not enough visible COLMAP points for this frame
        # Fall back to returning input as-is (will be wrong scale but won't crash)
        return d_pred

    # Project to image plane
    pts_proj = (K @ pts_cam.T).T           # [M, 3]
    u = pts_proj[:, 0] / pts_proj[:, 2]   # [M]
    v = pts_proj[:, 1] / pts_proj[:, 2]   # [M]
    z = pts_proj[:, 2]                    # [M]  metric depth

    # Keep only points within image bounds
    in_bounds = (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)
    u = u[in_bounds].astype(int)
    v = v[in_bounds].astype(int)
    z = z[in_bounds]

    if len(z) < 10:
        return d_pred

    # Sample predicted depth at projected pixel locations
    d_pred_np = d_pred.cpu().float().numpy()
    d_at_pts = d_pred_np[v, u]   # [M'] predicted relative depth at sparse pts

    # Least-squares fit
    s, t = least_squares_scale_shift(d_at_pts, z)

    d_aligned = s * d_pred + t
    d_aligned = d_aligned.clamp(min=0.0)
    return d_aligned


# ---------------------------------------------------------------------------
# Batch alignment (for evaluation over the full dataset)
# ---------------------------------------------------------------------------

def align_depth_batch(
    d_pred_batch: torch.Tensor,
    d_gt_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Align each frame in a batch independently.

    Parameters
    ----------
    d_pred_batch : [N, H, W]  predicted relative depth
    d_gt_batch   : [N, H, W]  GT metric depth

    Returns
    -------
    aligned : [N, H, W]  metric depth
    """
    aligned = []
    for i in range(d_pred_batch.shape[0]):
        a = align_depth_to_gt(d_pred_batch[i], d_gt_batch[i])
        aligned.append(a)
    return torch.stack(aligned, dim=0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    H, W = 64, 80

    # Simulate: true metric depth in [1, 5] metres
    d_gt = torch.rand(H, W) * 4 + 1.0
    # Simulate: relative depth with unknown scale 2.5 and shift 0.3
    d_rel = (d_gt - 0.3) / 2.5 + torch.randn(H, W) * 0.02
    mask = d_gt > 0

    d_aligned = align_depth_to_gt(d_rel, d_gt, mask)

    err = (d_aligned - d_gt).abs()
    print(f"Mean alignment error: {err.mean():.4f} m  (should be near 0)")
    print(f"Max alignment error : {err.max():.4f} m")
    print("Scale alignment test passed.")
