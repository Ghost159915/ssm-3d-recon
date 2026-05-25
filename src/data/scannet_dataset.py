"""
src/data/scannet_dataset.py
===========================
Dataset loader for ScanNet v2 scenes.

ScanNet is the gold-standard indoor RGB-D dataset: 1513 scenes of apartments,
offices, kitchens, living rooms, and more — captured with a Structure Sensor
depth camera. Each scene contains 500–5000 frames with metric depth and
BundleFusion-estimated camera poses.

Download
--------
1. Request access at https://www.scan-net.org/  (takes 1–2 days to approve)
2. Once approved, download the ScanNet downloader script and run:

    python download-scannet.py -o data/scannet --type .sens
    python download-scannet.py -o data/scannet --type _vh_clean_2.labels.ply

   Or for a specific scene (faster for testing):
    python download-scannet.py -o data/scannet --id scene0000_00 --type .sens

3. Extract .sens files with the SensReader tool:
    cd ScanNet/SensReader/python
    python reader.py --filename data/scannet/scene0000_00/scene0000_00.sens \
                     --output_path data/scannet/scene0000_00 \
                     --export_color_images --export_depth_images \
                     --export_poses --export_intrinsics

After extraction, each scene folder looks like:
    data/scannet/scene0000_00/
    ├── color/          ← 000000.jpg, 000001.jpg, ...
    ├── depth/          ← 000000.png, 000001.png, ... (uint16 mm → /1000 = metres)
    ├── pose/           ← 000000.txt, 000001.txt, ... (4×4 cam-to-world)
    └── intrinsic/
        └── intrinsic_color.txt   (4×4 matrix, top-left 3×3 = K)

Good scenes to start with (diverse, well-reconstructed):
    scene0000_00  — office/lab
    scene0001_00  — bedroom
    scene0002_00  — kitchen
    scene0003_00  — bathroom
    scene0004_00  — living room
    scene0005_00  — conference room

Usage
-----
    from src.data.scannet_dataset import ScanNetDataset
    ds = ScanNetDataset("data/scannet/scene0000_00")
    sample = ds[0]
    # same keys as TUMDataset: rgb, depth, pose, K, ts, rgb_path, depth_path
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ScanNet depth images are uint16 in millimetres
DEPTH_SCALE = 1000.0
# Depth values above this (in metres) are invalid
MAX_DEPTH_M = 10.0


def _load_pose(path: str) -> np.ndarray:
    """Load a 4×4 cam-to-world matrix from a ScanNet pose .txt file."""
    with open(path) as f:
        vals = []
        for line in f:
            line = line.strip()
            if line:
                vals.extend(map(float, line.split()))
    mat = np.array(vals, dtype=np.float64).reshape(4, 4)
    return mat


def _load_intrinsics(path: str) -> np.ndarray:
    """Load the 3×3 intrinsic matrix from ScanNet's intrinsic_color.txt."""
    with open(path) as f:
        vals = []
        for line in f:
            line = line.strip()
            if line:
                vals.extend(map(float, line.split()))
    mat4 = np.array(vals, dtype=np.float64).reshape(4, 4)
    return mat4[:3, :3]   # top-left 3×3 is K


class ScanNetDataset(Dataset):
    """
    ScanNet v2 scene dataset.

    Parameters
    ----------
    root       : path to an extracted ScanNet scene folder
                 (must contain color/, depth/, pose/, intrinsic/)
    max_frames : limit number of frames (None = all)
    img_size   : optional (H, W) to resize frames
    stride     : step between frames (1 = every frame, 5 = every 5th)
                 Useful for long scenes — ScanNet records at 25 fps so
                 stride=5 gives ~5 fps equivalent.
    max_depth  : depth values above this metres are set to 0 (invalid)
    """

    def __init__(
        self,
        root: str,
        max_frames: Optional[int] = None,
        img_size: Optional[Tuple[int, int]] = None,
        stride: int = 5,
        max_depth: float = MAX_DEPTH_M,
    ):
        self.root      = Path(root).expanduser().resolve()
        self.img_size  = img_size
        self.max_depth = max_depth

        # Validate structure
        for sub in ("color", "depth", "pose"):
            if not (self.root / sub).exists():
                raise FileNotFoundError(
                    f"'{sub}/' not found in {self.root}. "
                    "Run SensReader to extract the .sens file first.\n"
                    "See the docstring in src/data/scannet_dataset.py for instructions."
                )

        # Intrinsics
        K_path = self.root / "intrinsic" / "intrinsic_color.txt"
        if not K_path.exists():
            # Fallback: look for intrinsic_depth.txt
            K_path = self.root / "intrinsic" / "intrinsic_depth.txt"
        self.K = _load_intrinsics(str(K_path))

        # Discover frames
        color_dir = self.root / "color"
        color_files = sorted(color_dir.glob("*.jpg")) + sorted(color_dir.glob("*.png"))
        color_files = sorted(color_files, key=lambda p: int(p.stem))

        depth_dir = self.root / "depth"
        pose_dir  = self.root / "pose"

        # Apply stride
        color_files = color_files[::stride]

        self.frames: List[Dict] = []
        for cf in color_files:
            idx = int(cf.stem)
            dp  = depth_dir / f"{idx:06d}.png"
            pp  = pose_dir  / f"{idx:06d}.txt"

            if not dp.exists():
                # Try without zero-padding
                dp = depth_dir / f"{idx}.png"
            if not pp.exists():
                pp = pose_dir / f"{idx}.txt"

            if not dp.exists() or not pp.exists():
                continue   # skip frames with missing data

            # Skip frames where pose is invalid (ScanNet marks these with inf)
            try:
                pose = _load_pose(str(pp))
                if not np.isfinite(pose).all():
                    continue
            except Exception:
                continue

            self.frames.append({
                "ts":         float(idx),
                "rgb_path":   str(cf),
                "depth_path": str(dp),
                "pose":       pose,
            })

        if max_frames is not None:
            self.frames = self.frames[:max_frames]

        print(f"[ScanNetDataset] {self.root.name}: {len(self)} frames "
              f"(stride={stride}, ScanNet indoor)")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.frames[idx]

        # RGB
        rgb_bgr = cv2.imread(entry["rgb_path"])
        if rgb_bgr is None:
            raise FileNotFoundError(f"RGB not found: {entry['rgb_path']}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # Depth (uint16 millimetres → float32 metres)
        d_raw = cv2.imread(entry["depth_path"], cv2.IMREAD_UNCHANGED)
        if d_raw is None:
            raise FileNotFoundError(f"Depth not found: {entry['depth_path']}")
        depth = d_raw.astype(np.float32) / DEPTH_SCALE
        depth[depth > self.max_depth] = 0.0

        # Resize
        if self.img_size is not None:
            H, W = self.img_size
            rgb   = cv2.resize(rgb,   (W, H), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        rgb_t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        pose_t  = torch.from_numpy(entry["pose"])
        K_t     = torch.from_numpy(self.K)

        return {
            "rgb":        rgb_t,
            "depth":      depth_t,
            "pose":       pose_t,
            "K":          K_t,
            "ts":         entry["ts"],
            "rgb_path":   entry["rgb_path"],
            "depth_path": entry["depth_path"],
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/scannet/scene0000_00"
    ds = ScanNetDataset(root, max_frames=5)
    s  = ds[0]
    print(f"RGB:   {s['rgb'].shape}  {s['rgb'].dtype}")
    print(f"Depth: {s['depth'].shape}  "
          f"min={s['depth'].min():.3f}  max={s['depth'].max():.3f}  "
          f"valid%={(s['depth'] > 0).float().mean()*100:.1f}%")
    print(f"Pose:\n{s['pose']}")
    print(f"K:\n{s['K']}")
