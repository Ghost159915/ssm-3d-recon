"""
src/data/arkitscenes_dataset.py
================================
Dataset loader for Apple ARKitScenes.

ARKitScenes is THE closest public dataset to your iPhone recording:
  - Captured with iPad Pro (same ARKit stack as iPhone)
  - LiDAR depth (metric, same sensor family as Pro iPhones)
  - ARKit camera poses
  - 5,047 indoor scenes: living rooms, bedrooms, kitchens, offices, bathrooms
  - Huge diversity — this is what will generalise your model to real phone video

Download (no registration required)
------------------------------------
Option A — HuggingFace CLI (recommended):
    pip install huggingface_hub
    python - <<'EOF'
    from huggingface_hub import snapshot_download
    # Download a handful of scenes for quick start
    snapshot_download(
        repo_id="apple/ARKitScenes",
        repo_type="dataset",
        allow_patterns=["*Training/4199*", "*Training/4204*", "*Training/4175*",
                        "*Training/4178*", "*Training/4181*"],
        local_dir="data/arkitscenes"
    )
    EOF

Option B — Download the full low-res split (~40 GB):
    python - <<'EOF'
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="apple/ARKitScenes",
        repo_type="dataset",
        allow_patterns=["*lowres*"],
        local_dir="data/arkitscenes"
    )
    EOF

Option C — Manual scene list download script (see scripts/download_arkitscenes.py)

Folder structure per scene (lowres split):
    data/arkitscenes/Training/4199/
    ├── lowres_wide/                ← RGB frames ~256×192 as {timestamp}.png
    ├── lowres_depth/               ← LiDAR depth {timestamp}.png (uint16 mm → /1000 m)
    ├── lowres_wide.traj            ← camera poses (one per line, see below)
    └── lowres_wide_intrinsics/     ← per-frame intrinsics as {timestamp}.plist

Trajectory file format (lowres_wide.traj):
    Each line: timestamp r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz 0 0 0 1
    (row-major 4×4 camera-to-world matrix, metric metres)

Usage
-----
    from src.data.arkitscenes_dataset import ARKitScenesDataset
    ds = ARKitScenesDataset("data/arkitscenes/Training/4199")
    sample = ds[0]
    # same keys as TUMDataset: rgb, depth, pose, K, ts, rgb_path, depth_path
"""

import plistlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DEPTH_SCALE  = 1000.0    # uint16 millimetres → metres
MAX_DEPTH_M  = 5.0       # LiDAR saturates around 5 m
MIN_DEPTH_M  = 0.1       # minimum reliable LiDAR range


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def _parse_traj(traj_path: str) -> Dict[float, np.ndarray]:
    """
    Parse ARKitScenes .traj file.

    Each line: timestamp followed by 16 space-separated floats (row-major
    4×4 camera-to-world SE(3) matrix, already in metric metres).

    Returns dict: timestamp (float) → pose [4, 4] float64.
    """
    poses = {}
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 17:
                continue
            ts   = float(parts[0])
            vals = list(map(float, parts[1:17]))
            mat  = np.array(vals, dtype=np.float64).reshape(4, 4)
            # Some frames have NaN/inf (tracking lost) — skip them
            if np.isfinite(mat).all():
                poses[ts] = mat
    return poses


def _nearest_ts(query: float, candidates: List[float], max_diff: float = 0.05):
    """Find the nearest timestamp in candidates to query."""
    if not candidates:
        return None
    idx  = np.searchsorted(candidates, query)
    best = None
    for ci in [idx - 1, idx]:
        if 0 <= ci < len(candidates):
            diff = abs(candidates[ci] - query)
            if diff <= max_diff:
                if best is None or diff < abs(candidates[best] - query):
                    best = ci
    return candidates[best] if best is not None else None


def _read_intrinsics_plist(path: str) -> np.ndarray:
    """
    Read ARKitScenes per-frame intrinsics from a .plist file.
    Returns 3×3 K matrix.
    """
    with open(path, "rb") as f:
        data = plistlib.load(f)
    # Keys: fx, fy, cx, cy  (sometimes as 'intrinsic_matrix' 3x3 col-major)
    if "intrinsic_matrix" in data:
        # 9-element list, column-major → transpose
        m = np.array(data["intrinsic_matrix"], dtype=np.float64).reshape(3, 3).T
        return m
    fx = float(data.get("fx", data.get("focal_length_x", 0)))
    fy = float(data.get("fy", data.get("focal_length_y", fx)))
    cx = float(data.get("cx", data.get("principal_point_x", 0)))
    cy = float(data.get("cy", data.get("principal_point_y", 0)))
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ARKitScenesDataset(Dataset):
    """
    ARKitScenes low-res split dataset.

    Parameters
    ----------
    root       : path to a single ARKitScenes scene folder
                 (must contain lowres_wide/, lowres_depth/, lowres_wide.traj)
    max_frames : limit number of frames (None = all)
    img_size   : optional (H, W) to resize frames
    stride     : step between matched frames (1 = ~30 fps → slow; 3 = ~10 fps)
    max_depth  : LiDAR readings above this metres are set to 0 (invalid)
    """

    def __init__(
        self,
        root: str,
        max_frames: Optional[int] = None,
        img_size: Optional[Tuple[int, int]] = None,
        stride: int = 3,
        max_depth: float = MAX_DEPTH_M,
    ):
        self.root      = Path(root).expanduser().resolve()
        self.img_size  = img_size
        self.max_depth = max_depth

        # Validate
        rgb_dir   = self.root / "lowres_wide"
        depth_dir = self.root / "lowres_depth"
        traj_path = self.root / "lowres_wide.traj"
        intr_dir  = self.root / "lowres_wide_intrinsics"

        for p, label in [(rgb_dir, "lowres_wide/"),
                         (depth_dir, "lowres_depth/"),
                         (traj_path, "lowres_wide.traj")]:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"'{label}' not found in {self.root}.\n"
                    "See src/data/arkitscenes_dataset.py docstring for download instructions."
                )

        # Load poses
        pose_map = _parse_traj(str(traj_path))
        pose_ts  = sorted(pose_map.keys())

        # Discover RGB frames sorted by timestamp
        rgb_files = sorted(rgb_dir.glob("*.png"), key=lambda p: float(p.stem))
        rgb_files = rgb_files[::stride]

        # Intrinsics: use first available plist, keep fixed for the scene
        self.K = None
        if intr_dir.exists():
            plists = sorted(intr_dir.glob("*.plist"))
            if plists:
                try:
                    self.K = _read_intrinsics_plist(str(plists[0]))
                except Exception:
                    pass

        if self.K is None:
            # Fallback: typical ARKitScenes lowres intrinsics (~256×192)
            self.K = np.array([[211.0,   0.0, 128.0],
                               [  0.0, 211.0,  96.0],
                               [  0.0,   0.0,   1.0]], dtype=np.float64)
            print("[ARKitScenesDataset] Warning: no intrinsics found, using defaults")

        # Build frame list by matching RGB → depth → pose timestamps
        self.frames: List[Dict] = []
        for rf in rgb_files:
            rgb_ts = float(rf.stem)

            # Match depth
            dp = depth_dir / rf.name
            if not dp.exists():
                continue

            # Match pose (nearest timestamp within 50 ms)
            matched_ts = _nearest_ts(rgb_ts, pose_ts, max_diff=0.05)
            if matched_ts is None:
                continue

            self.frames.append({
                "ts":         rgb_ts,
                "rgb_path":   str(rf),
                "depth_path": str(dp),
                "pose":       pose_map[matched_ts],
            })

        if max_frames is not None:
            self.frames = self.frames[:max_frames]

        print(f"[ARKitScenesDataset] {self.root.name}: {len(self)} frames "
              f"(stride={stride}, iPad Pro LiDAR)")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.frames[idx]

        # RGB
        rgb_bgr = cv2.imread(entry["rgb_path"])
        if rgb_bgr is None:
            raise FileNotFoundError(f"RGB not found: {entry['rgb_path']}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # Depth (uint16 mm → float32 m)
        d_raw = cv2.imread(entry["depth_path"], cv2.IMREAD_UNCHANGED)
        if d_raw is None:
            H_r, W_r = rgb.shape[:2]
            depth = np.zeros((H_r, W_r), dtype=np.float32)
        else:
            depth = d_raw.astype(np.float32) / DEPTH_SCALE

        depth[depth > self.max_depth] = 0.0
        depth[depth < MIN_DEPTH_M]   = 0.0

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
    root = sys.argv[1] if len(sys.argv) > 1 else "data/arkitscenes/Training/4199"
    ds = ARKitScenesDataset(root, max_frames=5)
    s  = ds[0]
    print(f"RGB:   {s['rgb'].shape}  {s['rgb'].dtype}")
    print(f"Depth: {s['depth'].shape}  "
          f"min={s['depth'][s['depth']>0].min():.3f}  "
          f"max={s['depth'].max():.3f}  "
          f"valid%={(s['depth']>0).float().mean()*100:.1f}%")
    print(f"Pose:\n{s['pose']}")
    print(f"K:\n{s['K']}")
