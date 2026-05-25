"""
src/data/colmap_dataset.py
==========================
Dataset class for COLMAP-processed phone video.

Reads the output of scripts/00_process_video.py and returns the same
interface as TUMDataset — so all existing scripts work unchanged.

Folder structure expected:
    data/phone_scene/
    ├── frames/           PNG frames extracted from video
    ├── poses.npy         [N, 4, 4] camera-to-world SE(3), metric
    ├── intrinsics.npy    [3, 3] camera intrinsic matrix
    └── frame_names.txt   ordered list of frame filenames used by COLMAP

Usage:
    from src.data.colmap_dataset import ColmapDataset
    ds = ColmapDataset("data/phone_scene")
    sample = ds[0]
    # same keys as TUMDataset: rgb, depth, pose, K, ts, rgb_path
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ColmapDataset(Dataset):
    """
    Dataset for COLMAP-processed phone video.

    Parameters
    ----------
    root       : path to the processed scene folder
    max_frames : limit number of frames (None = all)
    img_size   : optional (H, W) to resize frames
    """

    def __init__(
        self,
        root: str,
        max_frames: Optional[int] = None,
        img_size: Optional[Tuple[int, int]] = None,
    ):
        self.root     = Path(root).expanduser().resolve()
        self.img_size = img_size

        # Load poses [N, 4, 4]
        poses_path = self.root / "poses.npy"
        if not poses_path.exists():
            raise FileNotFoundError(
                f"poses.npy not found in {self.root}. "
                "Run scripts/00_process_video.py first."
            )
        self.poses = np.load(str(poses_path))   # [N, 4, 4] float64

        # Load intrinsics [3, 3]
        K_path = self.root / "intrinsics.npy"
        self.K = np.load(str(K_path))           # [3, 3] float64

        # Load ordered frame names
        names_path = self.root / "frame_names.txt"
        with open(names_path) as f:
            frame_names = [l.strip() for l in f if l.strip()]

        frames_dir = self.root / "frames"
        self.frame_paths = [str(frames_dir / n) for n in frame_names]

        assert len(self.poses) == len(self.frame_paths), (
            f"poses ({len(self.poses)}) and frames ({len(self.frame_paths)}) mismatch"
        )

        if max_frames is not None:
            self.poses       = self.poses[:max_frames]
            self.frame_paths = self.frame_paths[:max_frames]

        print(f"[ColmapDataset] {self.root.name}: {len(self)} frames "
              f"(phone video, no GT depth)")

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.frame_paths[idx]

        rgb_bgr = cv2.imread(path)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)   # [H, W, 3] uint8

        if self.img_size is not None:
            H, W = self.img_size
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        # No GT depth for phone video — return zeros (signals "no GT")
        H_img, W_img = rgb.shape[:2]
        depth = np.zeros((H_img, W_img), dtype=np.float32)

        rgb_t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        pose_t  = torch.from_numpy(self.poses[idx])
        K_t     = torch.from_numpy(self.K)

        return {
            "rgb":        rgb_t,       # [3, H, W] float32
            "depth":      depth_t,     # [1, H, W] zeros (no GT)
            "pose":       pose_t,      # [4, 4] float64 cam-to-world metric
            "K":          K_t,         # [3, 3] float64
            "ts":         float(idx),
            "rgb_path":   path,
            "depth_path": "",
        }
