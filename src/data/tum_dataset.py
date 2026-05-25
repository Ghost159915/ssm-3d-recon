"""
src/data/tum_dataset.py
=======================
TUM RGB-D benchmark data loader.

Handles:
  - Timestamp association between rgb.txt / depth.txt / groundtruth.txt
    (files are not perfectly synchronised — we use nearest-neighbour
     matching within a configurable tolerance, same as the official
     TUM evaluation scripts)
  - Depth loading (uint16 PNG → float32 metres via /5000.0)
  - Ground-truth pose loading (quaternion → 4×4 SE(3) matrix)
  - PyTorch Dataset interface; returns dicts with tensors

Usage
-----
    from src.data.tum_dataset import TUMDataset

    ds = TUMDataset("data/rgbd_dataset_freiburg1_desk")
    print(len(ds))          # number of associated frames

    sample = ds[0]
    # sample["rgb"]      — torch.float32  [3, H, W]  range [0,1]
    # sample["depth"]    — torch.float32  [1, H, W]  metres (0 = invalid)
    # sample["pose"]     — torch.float64  [4, 4]     camera-to-world SE(3)
    # sample["K"]        — torch.float64  [3, 3]     intrinsic matrix
    # sample["ts"]       — float          RGB timestamp (seconds)
    # sample["rgb_path"] — str            absolute path to RGB image
    # sample["depth_path"] — str          absolute path to depth image

Smoke test
----------
    python -c "
    from src.data.tum_dataset import TUMDataset
    ds = TUMDataset('data/rgbd_dataset_freiburg1_desk')
    print(f'Frames: {len(ds)}')
    s = ds[0]
    print('rgb:', s['rgb'].shape, s['rgb'].dtype)
    print('depth:', s['depth'].shape, 'min/max:', s['depth'].min().item(), s['depth'].max().item())
    print('pose:', s['pose'])
    "
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Camera intrinsics for each TUM freiburg sequence
# ---------------------------------------------------------------------------
INTRINSICS = {
    "freiburg1": dict(fx=517.3, fy=516.5, cx=318.6, cy=255.3),
    "freiburg2": dict(fx=520.9, fy=521.0, cx=325.1, cy=249.7),
    "freiburg3": dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6),
}

# Depth images are uint16; divide by this to get metres
DEPTH_SCALE = 5000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tum_timestamps(filepath: str) -> List[Tuple[float, str]]:
    """
    Read a TUM-format list file (rgb.txt / depth.txt / groundtruth.txt).
    Returns list of (timestamp_float, rest_of_line_string).
    Lines starting with '#' are skipped.
    """
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = float(parts[0])
            rest = " ".join(parts[1:])
            entries.append((ts, rest))
    return entries


def _associate(
    list_a: List[Tuple[float, str]],
    list_b: List[Tuple[float, str]],
    max_diff: float = 0.02,
) -> List[Tuple[int, int]]:
    """
    Nearest-neighbour timestamp association between two sorted lists.
    Returns list of (idx_a, idx_b) pairs whose timestamps differ by at most
    max_diff seconds. Identical to the TUM benchmark evaluation approach.
    """
    # Build a sorted array of timestamps for list_b for fast search
    ts_b = np.array([t for t, _ in list_b])
    matches = []
    for i, (ts_a, _) in enumerate(list_a):
        j = int(np.searchsorted(ts_b, ts_a))
        best = None
        for candidate in [j - 1, j]:
            if 0 <= candidate < len(ts_b):
                diff = abs(ts_b[candidate] - ts_a)
                if diff <= max_diff:
                    if best is None or diff < abs(ts_b[best] - ts_a):
                        best = candidate
        if best is not None:
            matches.append((i, best))
    return matches


def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert a unit quaternion (qx, qy, qz, qw) to a 3×3 rotation matrix.
    Uses the standard formula; no external dependencies.
    """
    # Normalise (defensive)
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    return R


def _pose_to_se3(tx: float, ty: float, tz: float,
                 qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Build a 4×4 camera-to-world SE(3) matrix from TUM ground-truth format:
        timestamp tx ty tz qx qy qz qw   (world-to-camera in TUM)

    TUM groundtruth.txt stores the transform from camera to world, i.e.
    p_world = T @ p_cam.  So this matrix IS already camera-to-world.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_rotation_matrix(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


def _detect_sequence_type(root: str) -> str:
    """Detect freiburg1/2/3 from the folder name."""
    root_lower = str(root).lower()
    for seq in ("freiburg1", "freiburg2", "freiburg3"):
        if seq in root_lower:
            return seq
    # Default to freiburg1 with a warning
    print(f"[TUMDataset] Warning: could not detect sequence type from '{root}'. "
          "Defaulting to freiburg1 intrinsics.")
    return "freiburg1"


def _build_intrinsic_matrix(seq_type: str) -> np.ndarray:
    intr = INTRINSICS[seq_type]
    K = np.array([
        [intr["fx"], 0.0,        intr["cx"]],
        [0.0,        intr["fy"], intr["cy"]],
        [0.0,        0.0,        1.0       ],
    ], dtype=np.float64)
    return K


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TUMDataset(Dataset):
    """
    TUM RGB-D benchmark dataset loader.

    Parameters
    ----------
    root : str | Path
        Path to the extracted TUM scene folder, e.g.
        'data/rgbd_dataset_freiburg1_desk'
    max_frames : int | None
        If set, only load the first N associated frames. Handy for quick
        debugging without iterating the full ~600-frame sequence.
    img_size : tuple | None
        If (H, W), resize RGB and depth to this resolution. None = original
        640×480.
    max_depth : float
        Depth values above this (metres) are set to 0 (invalid). TUM sensor
        saturates around 8 m; 10.0 is a safe ceiling.
    association_max_diff : float
        Maximum timestamp difference (seconds) for RGB/depth/pose association.
    """

    def __init__(
        self,
        root: str,
        max_frames: Optional[int] = None,
        img_size: Optional[Tuple[int, int]] = None,
        max_depth: float = 10.0,
        association_max_diff: float = 0.02,
    ):
        self.root = Path(root).expanduser().resolve()
        self.img_size = img_size
        self.max_depth = max_depth

        # Detect intrinsics
        seq_type = _detect_sequence_type(str(self.root))
        self.K = _build_intrinsic_matrix(seq_type)

        # Parse file lists
        rgb_list  = _parse_tum_timestamps(str(self.root / "rgb.txt"))
        depth_list = _parse_tum_timestamps(str(self.root / "depth.txt"))
        gt_list   = _parse_tum_timestamps(str(self.root / "groundtruth.txt"))

        # Associate: first rgb↔depth, then result↔groundtruth
        rgb_depth_matches = _associate(rgb_list, depth_list, association_max_diff)
        # Build intermediate list aligned by rgb index
        rd_rgb   = [(rgb_list[i][0],   rgb_list[i][1])   for i, _ in rgb_depth_matches]
        rd_depth = [(depth_list[j][0], depth_list[j][1]) for _, j in rgb_depth_matches]

        rd_gt_matches = _associate(rd_rgb, gt_list, association_max_diff)

        # Final triple-associated entries
        self.frames: List[Dict] = []
        for idx_rd, idx_gt in rd_gt_matches:
            ts_rgb, rgb_file   = rd_rgb[idx_rd]
            ts_depth, dep_file = rd_depth[idx_rd]
            ts_gt, gt_data     = gt_list[idx_gt]

            # Parse pose: "tx ty tz qx qy qz qw"
            vals = list(map(float, gt_data.split()))
            tx, ty, tz, qx, qy, qz, qw = vals
            pose = _pose_to_se3(tx, ty, tz, qx, qy, qz, qw)

            self.frames.append({
                "ts":         ts_rgb,
                "rgb_path":   str(self.root / rgb_file),
                "depth_path": str(self.root / dep_file),
                "pose":       pose,        # numpy [4,4]
            })

        if max_frames is not None:
            self.frames = self.frames[:max_frames]

        print(f"[TUMDataset] {self.root.name}: {len(self.frames)} associated frames "
              f"(seq={seq_type})")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.frames[idx]

        # --- Load RGB ---
        rgb_bgr = cv2.imread(entry["rgb_path"])
        if rgb_bgr is None:
            raise FileNotFoundError(f"RGB image not found: {entry['rgb_path']}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)  # [H, W, 3] uint8

        # --- Load depth ---
        depth_raw = cv2.imread(entry["depth_path"], cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Depth image not found: {entry['depth_path']}")
        depth = depth_raw.astype(np.float32) / DEPTH_SCALE   # metres

        # --- Optional resize ---
        if self.img_size is not None:
            H, W = self.img_size
            rgb   = cv2.resize(rgb,   (W, H), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        # --- Clip invalid depth ---
        depth[depth > self.max_depth] = 0.0   # 0 = invalid (sensor max)

        # --- Convert to tensors ---
        # RGB: [H,W,3] uint8 → [3,H,W] float32 [0,1]
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        # Depth: [H,W] → [1,H,W] float32
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        # Pose: numpy [4,4] → torch float64
        pose_t = torch.from_numpy(entry["pose"])
        # Intrinsics
        K_t = torch.from_numpy(self.K)

        return {
            "rgb":        rgb_t,          # [3, H, W]  float32
            "depth":      depth_t,        # [1, H, W]  float32, metres
            "pose":       pose_t,         # [4, 4]     float64, cam-to-world
            "K":          K_t,            # [3, 3]     float64
            "ts":         entry["ts"],    # float
            "rgb_path":   entry["rgb_path"],
            "depth_path": entry["depth_path"],
        }

    # ------------------------------------------------------------------
    def get_sequence(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Return a batched dict of tensors for frames [start:end:step].
        Useful for feeding a full sequence to the S5 temporal module.

            rgb   — [T, 3, H, W]
            depth — [T, 1, H, W]
            pose  — [T, 4, 4]
            K     — [3, 3]   (same for all frames)
        """
        indices = range(start, end or len(self), step)
        samples = [self[i] for i in indices]

        return {
            "rgb":   torch.stack([s["rgb"]   for s in samples]),
            "depth": torch.stack([s["depth"] for s in samples]),
            "pose":  torch.stack([s["pose"]  for s in samples]),
            "K":     samples[0]["K"],
            "ts":    [s["ts"] for s in samples],
        }


# ---------------------------------------------------------------------------
# Smoke test (run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    root = sys.argv[1] if len(sys.argv) > 1 else "data/rgbd_dataset_freiburg1_desk"
    ds = TUMDataset(root, max_frames=10)

    print(f"\nDataset length : {len(ds)}")
    s = ds[0]
    print(f"RGB shape      : {s['rgb'].shape}   dtype={s['rgb'].dtype}")
    print(f"Depth shape    : {s['depth'].shape}  dtype={s['depth'].dtype}")
    print(f"Depth range    : {s['depth'].min():.3f} – {s['depth'].max():.3f} m")
    print(f"Pose (cam→world):\n{s['pose']}")
    print(f"Intrinsics K   :\n{s['K']}")

    # Quick visualisation
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(s["rgb"].permute(1, 2, 0).numpy())
    axes[0].set_title("RGB frame 0")
    axes[0].axis("off")
    axes[1].imshow(s["depth"].squeeze().numpy(), cmap="plasma")
    axes[1].set_title("Depth frame 0 (metres)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("smoke_test_frame0.png", dpi=120)
    print("\nSaved visualisation → smoke_test_frame0.png")
