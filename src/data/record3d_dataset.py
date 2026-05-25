"""
src/data/record3d_dataset.py
============================
Dataset loader for Record3D / Stray Scanner exports.

Both apps record iPhone LiDAR + RGB + ARKit poses simultaneously and export
them as a self-contained folder.  The folder structure differs slightly
between the two apps; this loader handles both.

--------------------------------------------------------------------
Record3D export structure (export → "Images + Depth + Poses"):
    scene/
    ├── rgbd/
    │   ├── 0.jpg (or .png)
    │   ├── 0.depth      ← raw float32 binary (H×W, little-endian)
    │   ├── 1.jpg
    │   ├── 1.depth
    │   └── ...
    └── metadata          ← JSON with intrinsics + poses

Stray Scanner export structure:
    scene/
    ├── color/
    │   ├── 000000.jpg
    │   └── ...
    ├── depth/
    │   ├── 000000.png   ← uint16, divide by 1000 → metres
    │   └── ...
    ├── camera_matrix.csv   ← fx,fy,cx,cy
    └── odometry.csv        ← tx ty tz qx qy qz qw per frame

Returns the same dict interface as TUMDataset:
    rgb        — [3, H, W]  float32 [0,1]
    depth      — [1, H, W]  float32 metres (0 = invalid / out-of-range)
    pose       — [4, 4]     float64 cam-to-world SE(3)
    K          — [3, 3]     float64
    ts         — float
    rgb_path   — str
    depth_path — str

Usage:
    from src.data.record3d_dataset import Record3DDataset
    ds = Record3DDataset("data/my_record3d_export")
    sample = ds[0]
"""

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Quaternion → 4×4 SE(3)  (ARKit convention: qw last in most exports)
# ---------------------------------------------------------------------------

def _quat_wxyz_to_se3(tx, ty, tz, qw, qx, qy, qz) -> np.ndarray:
    """Build cam-to-world SE(3) from ARKit translation + quaternion (qw first)."""
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = [tx, ty, tz]
    return T


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(root: Path) -> str:
    """Return 'record3d' or 'stray'."""
    if (root / "metadata").exists() and (root / "rgbd").exists():
        return "record3d"
    if (root / "odometry.csv").exists() and (root / "color").exists():
        return "stray"
    raise FileNotFoundError(
        f"Cannot detect Record3D or Stray Scanner format in {root}.\n"
        "Expected either:\n"
        "  Record3D: metadata + rgbd/ folder\n"
        "  Stray Scanner: odometry.csv + color/ + depth/ folders"
    )


# ---------------------------------------------------------------------------
# Record3D parser
# ---------------------------------------------------------------------------

def _load_record3d(root: Path, max_frames: Optional[int], img_size) -> Tuple:
    """Parse a Record3D export. Returns (frame_paths, depth_paths, poses, K)."""
    meta_path = root / "metadata"
    with open(meta_path) as f:
        meta = json.load(f)

    # Intrinsics
    K_data = meta["K"]   # [fx, 0, cx, 0, fy, cy, 0, 0, 1] row-major
    K = np.array(K_data, dtype=np.float64).reshape(3, 3)

    # Poses: list of 16-element row-major 4×4 matrices (cam-to-world)
    raw_poses = meta["poses"]   # list of lists or flat list

    # RGB images sorted by frame index
    rgbd_dir = root / "rgbd"
    indices  = sorted({int(p.stem) for p in rgbd_dir.iterdir()
                       if p.suffix in (".jpg", ".png", ".jpeg")})

    if max_frames is not None:
        indices = indices[:max_frames]

    frame_paths = []
    depth_paths = []
    poses       = []

    for idx in indices:
        # Find RGB file
        for ext in (".jpg", ".png", ".jpeg"):
            p = rgbd_dir / f"{idx}{ext}"
            if p.exists():
                frame_paths.append(str(p))
                break
        else:
            continue

        depth_paths.append(str(rgbd_dir / f"{idx}.depth"))

        # Record3D pose: flat 16-float list, row-major
        pose_flat = raw_poses[idx]
        pose = np.array(pose_flat, dtype=np.float64).reshape(4, 4)
        poses.append(pose)

    return frame_paths, depth_paths, poses, K


def _load_record3d_depth(path: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Load a Record3D .depth file.
    Format: raw little-endian float32 blob, length = H * W.
    """
    raw = Path(path).read_bytes()
    H, W = shape
    n_expected = H * W
    if len(raw) == n_expected * 4:
        depth = np.frombuffer(raw, dtype="<f4").reshape(H, W).copy()
    elif len(raw) == n_expected * 2:
        # some versions: uint16 millimetres
        depth = np.frombuffer(raw, dtype="<u2").reshape(H, W).astype(np.float32) / 1000.0
    else:
        # Unknown — return zeros
        depth = np.zeros((H, W), dtype=np.float32)
    return depth


# ---------------------------------------------------------------------------
# Stray Scanner parser
# ---------------------------------------------------------------------------

def _load_stray(root: Path, max_frames: Optional[int], img_size) -> Tuple:
    """Parse a Stray Scanner export. Returns (frame_paths, depth_paths, poses, K)."""
    # Intrinsics
    cam_path = root / "camera_matrix.csv"
    with open(cam_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    # Format: fx,0,cx / 0,fy,cy / 0,0,1   (3 lines, comma-separated)
    rows = [[float(v) for v in l.split(",")] for l in lines]
    K = np.array(rows, dtype=np.float64)

    # Poses
    odo_path = root / "odometry.csv"
    with open(odo_path) as f:
        odo_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    color_dir = root / "color"
    depth_dir = root / "depth"
    rgb_files  = sorted(color_dir.glob("*.jpg")) + sorted(color_dir.glob("*.png"))
    rgb_files  = sorted(rgb_files, key=lambda p: int(p.stem))
    depth_files = sorted(depth_dir.glob("*.png"), key=lambda p: int(p.stem))

    n = min(len(rgb_files), len(depth_files), len(odo_lines))
    if max_frames is not None:
        n = min(n, max_frames)

    frame_paths = [str(rgb_files[i])   for i in range(n)]
    depth_paths = [str(depth_files[i]) for i in range(n)]
    poses = []

    for line in odo_lines[:n]:
        vals = list(map(float, line.split(",")))
        # tx ty tz qx qy qz qw  (ARKit: qw last)
        tx, ty, tz, qx, qy, qz, qw = vals[:7]
        poses.append(_quat_wxyz_to_se3(tx, ty, tz, qw, qx, qy, qz))

    return frame_paths, depth_paths, poses, K


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Record3DDataset(Dataset):
    """
    Dataset for iPhone LiDAR captures exported via Record3D or Stray Scanner.

    Parameters
    ----------
    root       : path to the exported scene folder
    max_frames : limit number of frames (None = all)
    img_size   : optional (H, W) to resize frames
    max_depth  : LiDAR depth values above this (metres) are set to 0 (invalid).
                 iPhone LiDAR saturates around 5 m; 8.0 is a safe ceiling.
    """

    LIDAR_DEPTH_SCALE_STRAY = 1000.0   # uint16 PNG → metres

    def __init__(
        self,
        root: str,
        max_frames: Optional[int] = None,
        img_size: Optional[Tuple[int, int]] = None,
        max_depth: float = 8.0,
    ):
        self.root      = Path(root).expanduser().resolve()
        self.img_size  = img_size
        self.max_depth = max_depth

        fmt = _detect_format(self.root)
        print(f"[Record3DDataset] Detected format: {fmt}")

        if fmt == "record3d":
            self.frame_paths, self.depth_paths, self.poses, self.K = \
                _load_record3d(self.root, max_frames, img_size)
            self._fmt = "record3d"
        else:
            self.frame_paths, self.depth_paths, self.poses, self.K = \
                _load_stray(self.root, max_frames, img_size)
            self._fmt = "stray"

        assert len(self.poses) == len(self.frame_paths), (
            f"poses ({len(self.poses)}) and frames ({len(self.frame_paths)}) mismatch"
        )

        print(f"[Record3DDataset] {self.root.name}: {len(self)} frames "
              f"(iPhone LiDAR, metric depth)")

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Dict:
        rgb_path   = self.frame_paths[idx]
        depth_path = self.depth_paths[idx]

        # --- RGB ---
        rgb_bgr = cv2.imread(rgb_path)
        if rgb_bgr is None:
            raise FileNotFoundError(f"RGB frame not found: {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        H_img, W_img = rgb.shape[:2]

        # --- Depth ---
        if self._fmt == "record3d":
            depth = _load_record3d_depth(depth_path, (H_img, W_img))
        else:
            # Stray Scanner: uint16 PNG in millimetres
            d_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if d_raw is None:
                depth = np.zeros((H_img, W_img), dtype=np.float32)
            else:
                depth = d_raw.astype(np.float32) / self.LIDAR_DEPTH_SCALE_STRAY

        # Invalidate out-of-range readings
        depth[depth > self.max_depth] = 0.0
        depth[depth < 0.05]           = 0.0

        # --- Optional resize ---
        if self.img_size is not None:
            H, W = self.img_size
            rgb   = cv2.resize(rgb,   (W, H), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        # --- Tensors ---
        rgb_t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        pose_t  = torch.from_numpy(self.poses[idx])
        K_t     = torch.from_numpy(self.K)

        return {
            "rgb":        rgb_t,        # [3, H, W]  float32
            "depth":      depth_t,      # [1, H, W]  float32 metres (0=invalid)
            "pose":       pose_t,       # [4, 4]     float64 cam-to-world
            "K":          K_t,          # [3, 3]     float64
            "ts":         float(idx),
            "rgb_path":   rgb_path,
            "depth_path": depth_path,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/record3d_scene"
    ds = Record3DDataset(root, max_frames=5)
    s  = ds[0]
    print(f"RGB:   {s['rgb'].shape}  {s['rgb'].dtype}")
    print(f"Depth: {s['depth'].shape}  min={s['depth'].min():.3f}  "
          f"max={s['depth'].max():.3f}  valid%={((s['depth']>0).float().mean()*100):.1f}%")
    print(f"Pose:\n{s['pose']}")
    print(f"K:\n{s['K']}")
