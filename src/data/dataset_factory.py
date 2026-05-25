"""
src/data/dataset_factory.py
============================
Auto-detect and instantiate the right dataset class from a folder path.

Detection order (first match wins):
  1. TUM RGB-D      — has rgb.txt + depth.txt + groundtruth.txt
  2. ScanNet        — has color/ + depth/ + pose/ + intrinsic/
  3. ARKitScenes    — has lowres_wide/ + lowres_wide.traj
  4. Record3D       — has rgbd/ + metadata
  5. Stray Scanner  — has color/ + odometry.csv
  6. COLMAP (phone) — has poses.npy + intrinsics.npy + frame_names.txt

Usage
-----
    from src.data.dataset_factory import make_dataset

    ds = make_dataset("data/rgbd_dataset_freiburg1_desk")   # → TUMDataset
    ds = make_dataset("data/scannet/scene0000_00")           # → ScanNetDataset
    ds = make_dataset("data/arkitscenes/Training/4199")      # → ARKitScenesDataset
    ds = make_dataset("data/phone_scene")                    # → ColmapDataset

You can also force a type:
    ds = make_dataset("data/my_scene", dataset_type="scannet")
"""

from pathlib import Path
from typing import Optional, Tuple


SUPPORTED_TYPES = ("tum", "scannet", "arkitscenes", "record3d", "stray", "colmap")


def detect_dataset_type(root: str) -> str:
    """
    Detect the dataset format from folder contents.
    Returns one of: 'tum', 'scannet', 'arkitscenes', 'record3d', 'stray', 'colmap'.
    Raises ValueError if none detected.
    """
    p = Path(root).expanduser().resolve()

    # TUM: has rgb.txt, depth.txt, groundtruth.txt
    if (p / "rgb.txt").exists() and (p / "depth.txt").exists():
        return "tum"

    # ScanNet: color/ + depth/ + pose/ directories
    if (p / "color").is_dir() and (p / "depth").is_dir() and (p / "pose").is_dir():
        return "scannet"

    # ARKitScenes: lowres_wide/ folder + .traj file
    if (p / "lowres_wide").is_dir() and (p / "lowres_wide.traj").exists():
        return "arkitscenes"

    # Record3D: rgbd/ folder + metadata JSON
    if (p / "rgbd").is_dir() and (p / "metadata").exists():
        return "record3d"

    # Stray Scanner: color/ + odometry.csv
    if (p / "color").is_dir() and (p / "odometry.csv").exists():
        return "stray"

    # COLMAP phone video: poses.npy + intrinsics.npy + frame_names.txt
    if (p / "poses.npy").exists() and (p / "frame_names.txt").exists():
        return "colmap"

    raise ValueError(
        f"Cannot detect dataset type for '{root}'.\n"
        f"Supported formats: {SUPPORTED_TYPES}\n"
        "Pass dataset_type= explicitly if auto-detection fails."
    )


def make_dataset(
    root: str,
    dataset_type: Optional[str] = None,
    max_frames: Optional[int] = None,
    img_size: Optional[Tuple[int, int]] = None,
    stride: int = 1,
):
    """
    Instantiate the appropriate dataset for a given folder.

    Parameters
    ----------
    root         : path to the scene/sequence folder
    dataset_type : force type ('tum'|'scannet'|'arkitscenes'|'record3d'|
                              'stray'|'colmap'). None = auto-detect.
    max_frames   : pass through to dataset __init__
    img_size     : optional (H, W) resize
    stride       : for ScanNet and ARKitScenes (recorded at 25-30 fps, so
                   stride=5 or stride=3 gives ~5-10 fps equivalent).
                   Ignored for TUM (already ~30 fps but sparse matched).
    """
    if dataset_type is None:
        dataset_type = detect_dataset_type(root)

    dataset_type = dataset_type.lower()

    if dataset_type == "tum":
        from src.data.tum_dataset import TUMDataset
        return TUMDataset(root, max_frames=max_frames, img_size=img_size)

    elif dataset_type == "scannet":
        from src.data.scannet_dataset import ScanNetDataset
        return ScanNetDataset(root, max_frames=max_frames, img_size=img_size, stride=stride)

    elif dataset_type == "arkitscenes":
        from src.data.arkitscenes_dataset import ARKitScenesDataset
        return ARKitScenesDataset(root, max_frames=max_frames, img_size=img_size, stride=stride)

    elif dataset_type in ("record3d", "stray"):
        from src.data.record3d_dataset import Record3DDataset
        return Record3DDataset(root, max_frames=max_frames, img_size=img_size)

    elif dataset_type == "colmap":
        from src.data.colmap_dataset import ColmapDataset
        return ColmapDataset(root, max_frames=max_frames, img_size=img_size)

    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'. "
                         f"Choose from: {SUPPORTED_TYPES}")
