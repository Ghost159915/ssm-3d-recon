"""
scripts/00_process_video.py
===========================
Phone video → COLMAP poses → metric-scaled dataset ready for reconstruction.

Pipeline:
  1. Extract frames from video at target fps (ffmpeg)
  2. Run COLMAP feature extraction + matching + SfM
  3. Parse COLMAP sparse model (cameras, images, 3D points)
  4. Compute metric scale by aligning COLMAP depths to DAV2 predictions
  5. Save poses.npy, intrinsics.npy, frame_names.txt

After running this, use:
    python scripts/05_tsdf_fusion.py \
        --data data/phone_scene \
        --mode ssm \
        --checkpoint outputs/ssm_model_v3_multi/best_model.pt

Requirements:
    brew install colmap ffmpeg

Run:
    python scripts/00_process_video.py \
        --video my_room.mp4 \
        --out data/phone_scene \
        --fps 5 \
        --checkpoint outputs/ssm_model_v3_multi/best_model.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",      type=str, required=True,
                   help="Input video file (.mp4, .mov, etc.)")
    p.add_argument("--out",        type=str, required=True,
                   help="Output dataset directory")
    p.add_argument("--fps",        type=float, default=5,
                   help="Frames per second to extract (default 5)")
    p.add_argument("--max_frames", type=int, default=300,
                   help="Maximum frames to process")
    p.add_argument("--checkpoint", type=str,
                   default="outputs/ssm_model_v3_multi/best_model.pt",
                   help="SSM checkpoint for scale alignment")
    p.add_argument("--dav2_size",  type=str, default="small")
    p.add_argument("--device",     type=str, default="auto")
    p.add_argument("--skip_colmap", action="store_true",
                   help="Skip COLMAP if already run (just reparse)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Extract frames
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, out_dir: Path, fps: float, max_frames: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check ffmpeg
    result = subprocess.run(["ffmpeg", "-version"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")

    print(f"[Step 1] Extracting frames at {fps} fps...")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",          # high quality JPEG-like compression
        str(out_dir / "frame_%04d.png"),
        "-y"
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    frames = sorted(out_dir.glob("frame_*.png"))
    if len(frames) > max_frames:
        # Remove excess frames
        for f in frames[max_frames:]:
            f.unlink()
        frames = frames[:max_frames]

    print(f"  Extracted {len(frames)} frames → {out_dir}")
    return frames


# ---------------------------------------------------------------------------
# Step 2: Run COLMAP
# ---------------------------------------------------------------------------

def run_colmap(frames_dir: Path, colmap_dir: Path):
    # Check colmap
    result = subprocess.run(["colmap", "help"],
                            capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("colmap not found. Install: brew install colmap")

    db_path     = colmap_dir / "database.db"
    sparse_dir  = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    print("[Step 2] Running COLMAP feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path",    str(frames_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0",
    ], check=True, capture_output=True)

    print("[Step 2] Running COLMAP feature matching...")
    subprocess.run([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SequentialMatching.overlap", "10",
        "--SiftMatching.use_gpu", "0",
    ], check=True, capture_output=True)

    print("[Step 2] Running COLMAP sparse reconstruction (SfM)...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path",    str(frames_dir),
        "--output_path",   str(sparse_dir),
        "--Mapper.num_threads", "4",
    ], check=True, capture_output=True)

    # Find the largest sparse model
    models = sorted(sparse_dir.iterdir())
    if not models:
        raise RuntimeError(
            "COLMAP failed to reconstruct any sparse model. "
            "Try better lighting, slower camera movement, or more texture."
        )
    best = models[0]   # COLMAP outputs 0, 1, 2... sorted by size
    print(f"[Step 2] COLMAP done. Sparse model at {best}")
    return best


# ---------------------------------------------------------------------------
# Step 3: Parse COLMAP output
# ---------------------------------------------------------------------------

def parse_colmap_cameras(cameras_txt: Path):
    """Parse cameras.txt → intrinsic matrix K."""
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
            model = parts[1]
            if model == "PINHOLE":
                fx, fy, cx, cy = map(float, parts[4:8])
            elif model == "SIMPLE_PINHOLE":
                f_val = float(parts[4])
                fx = fy = f_val
                cx, cy = map(float, parts[5:7])
            elif model in ("RADIAL", "OPENCV"):
                fx = float(parts[4])
                fy = float(parts[5]) if model == "OPENCV" else fx
                cx, cy = map(float, parts[5:7]) if model == "RADIAL" else map(float, parts[6:8])
            else:
                raise ValueError(f"Unsupported camera model: {model}")
            break

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)
    return K


def quat_to_rotation(qw, qx, qy, qz):
    """COLMAP quaternion (qw first) → 3×3 rotation matrix."""
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ], dtype=np.float64)


def parse_colmap_images(images_txt: Path):
    """
    Parse images.txt → dict: image_name → (camera-to-world SE3 [4,4], colmap_depth_fn)

    COLMAP stores world-to-camera: p_cam = R @ p_world + t
    We invert to get camera-to-world: T_c2w = [R^T | -R^T @ t]
    """
    poses = {}   # name → (T_c2w [4,4], R_w2c [3,3], t_w2c [3])
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if len(parts) < 9:
            i += 1
            continue
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz      = map(float, parts[5:8])
        name            = parts[9]

        R_w2c = quat_to_rotation(qw, qx, qy, qz)
        t_w2c = np.array([tx, ty, tz])

        # Camera-to-world
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c

        T_c2w = np.eye(4, dtype=np.float64)
        T_c2w[:3, :3] = R_c2w
        T_c2w[:3,  3] = t_c2w

        poses[name] = (T_c2w, R_w2c, t_w2c)
        i += 2   # skip the POINTS2D line

    return poses


def parse_colmap_points(points3d_txt: Path):
    """Parse points3D.txt → list of (xyz, [(image_id, point2d_idx), ...])"""
    points = []
    with open(points3d_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            # Track: pairs of (IMAGE_ID, POINT2D_IDX) starting at index 8
            track = []
            for j in range(8, len(parts) - 1, 2):
                track.append((int(parts[j]), int(parts[j+1])))
            points.append((np.array([x, y, z]), track))
    return points


# ---------------------------------------------------------------------------
# Step 4: Metric scale from DAV2
# ---------------------------------------------------------------------------

def compute_metric_scale(
    poses_w2c: dict,   # name → (T_c2w, R_w2c, t_w2c)
    points3d: list,
    frames_dir: Path,
    K: np.ndarray,
    dav2,
    device: str,
    max_points: int = 2000,
) -> float:
    """
    Align COLMAP reconstruction to metric scale using DAV2 depth predictions.

    For each visible 3D point:
      colmap_depth = depth in camera frame from COLMAP (up-to-scale)
      dav2_depth   = DAV2 prediction at that pixel (approximately metric)
      ratio = dav2_depth / colmap_depth

    metric_scale = median(ratios)
    """
    from src.models.depth_anything import DepthAnythingV2
    import cv2 as _cv2

    print("[Step 4] Computing metric scale from DAV2 depth predictions...")

    ratios = []
    # Sample a subset of 3D points for efficiency
    sampled = points3d[:max_points] if len(points3d) > max_points else points3d

    # Cache DAV2 predictions per frame
    dav2_cache = {}

    for xyz, track in tqdm(sampled, desc="Scale alignment"):
        for img_id, pt2d_idx in track[:1]:   # use first observation per point
            # Find frame name for this image_id
            # COLMAP image_ids are 1-indexed
            pass

    # Simpler approach: iterate frame by frame, project all visible points
    # Build image_id → name mapping from poses dict
    # Actually we stored by name, so let's do it differently

    # For each frame, get DAV2 depth, project COLMAP points, collect ratios
    frame_names = sorted(poses_w2c.keys())
    name_to_idx = {n: i for i, n in enumerate(frame_names)}

    for name, (T_c2w, R_w2c, t_w2c) in list(poses_w2c.items())[:50]:
        img_path = frames_dir / name
        if not img_path.exists():
            continue

        if name not in dav2_cache:
            img_bgr = _cv2.imread(str(img_path))
            img_rgb = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]
            rgb_t = torch.from_numpy(img_rgb).permute(2,0,1).float()/255.

            with torch.no_grad():
                d_pred = dav2.predict(
                    rgb_t.unsqueeze(0).to(device),
                    output_size=(H, W)
                ).squeeze().cpu().numpy()
            dav2_cache[name] = (d_pred, H, W)

        d_pred, H, W = dav2_cache[name]
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        # Project 3D points into this camera
        for xyz, track in sampled:
            # Transform to camera frame
            p_cam = R_w2c @ xyz + t_w2c
            if p_cam[2] <= 0:
                continue   # behind camera

            colmap_depth = p_cam[2]

            # Project to pixel
            u = int(fx * p_cam[0] / p_cam[2] + cx + 0.5)
            v = int(fy * p_cam[1] / p_cam[2] + cy + 0.5)

            if not (0 <= u < W and 0 <= v < H):
                continue

            dav2_depth = float(d_pred[v, u])
            if dav2_depth < 0.1:
                continue

            ratios.append(dav2_depth / colmap_depth)

    if not ratios:
        print("  WARNING: no valid scale ratios — defaulting to scale=1.0")
        return 1.0

    scale = float(np.median(ratios))
    print(f"  Collected {len(ratios)} scale ratios")
    print(f"  Metric scale (median): {scale:.4f}  (std={float(np.std(ratios)):.3f})")
    return scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames_dir = out / "frames"
    colmap_dir = out / "colmap"

    if args.device == "auto":
        import torch
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device

    print(f"\n{'='*60}")
    print(f"  Video → 3D Dataset Pipeline")
    print(f"{'='*60}")
    print(f"  Input : {args.video}")
    print(f"  Output: {args.out}")
    print(f"  Device: {device}\n")

    # Step 1: Extract frames
    if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
        extract_frames(args.video, frames_dir, args.fps, args.max_frames)
    else:
        n = len(list(frames_dir.glob("*.png")))
        print(f"[Step 1] Frames already extracted: {n} frames in {frames_dir}")

    # Step 2: COLMAP
    sparse_model = None
    if not args.skip_colmap:
        sparse_model = run_colmap(frames_dir, colmap_dir)
    else:
        candidates = sorted((colmap_dir / "sparse").iterdir())
        sparse_model = candidates[0]
        print(f"[Step 2] Skipping COLMAP, using existing: {sparse_model}")

    # Step 3: Parse COLMAP
    print("[Step 3] Parsing COLMAP sparse model...")
    cameras_txt  = sparse_model / "cameras.txt"
    images_txt   = sparse_model / "images.txt"
    points3d_txt = sparse_model / "points3D.txt"

    # Try binary format if text not available
    if not cameras_txt.exists():
        print("  Text files not found, converting from binary...")
        subprocess.run([
            "colmap", "model_converter",
            "--input_path",  str(sparse_model),
            "--output_path", str(sparse_model),
            "--output_type", "TXT",
        ], check=True, capture_output=True)

    K            = parse_colmap_cameras(cameras_txt)
    poses_data   = parse_colmap_images(images_txt)
    points3d     = parse_colmap_points(points3d_txt)

    print(f"  Camera intrinsics:\n    fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"  Poses: {len(poses_data)} frames registered")
    print(f"  Sparse points: {len(points3d):,}")

    if len(poses_data) < 10:
        print("\nWARNING: Very few frames registered by COLMAP.")
        print("Tips for better results:")
        print("  - Move camera slowly and smoothly")
        print("  - Ensure good, even lighting")
        print("  - Keep textured surfaces in frame (avoid plain white walls)")

    # Step 4: Metric scale
    from src.models.depth_anything import DepthAnythingV2
    dav2 = DepthAnythingV2(size=args.dav2_size, device=device)
    metric_scale = compute_metric_scale(
        poses_data, points3d, frames_dir, K, dav2, device
    )
    del dav2

    # Step 5: Save dataset
    print("\n[Step 5] Saving dataset...")

    # Sort frames by name for temporal order
    sorted_names = sorted(poses_data.keys())
    T_c2w_list   = []

    for name in sorted_names:
        T_c2w, R_w2c, t_w2c = poses_data[name]
        # Apply metric scale to translation
        T_metric = T_c2w.copy()
        T_metric[:3, 3] *= metric_scale
        T_c2w_list.append(T_metric)

    poses_arr = np.stack(T_c2w_list, axis=0)   # [N, 4, 4]

    np.save(str(out / "poses.npy"),      poses_arr)
    np.save(str(out / "intrinsics.npy"), K)
    with open(out / "frame_names.txt", "w") as f:
        for name in sorted_names:
            f.write(name + "\n")
    with open(out / "metric_scale.txt", "w") as f:
        f.write(str(metric_scale))

    print(f"  poses.npy       → {poses_arr.shape}")
    print(f"  intrinsics.npy  → {K.shape}")
    print(f"  frame_names.txt → {len(sorted_names)} frames")
    print(f"  metric_scale    → {metric_scale:.4f}")

    print(f"\n{'='*60}")
    print(f"  Done! Dataset ready at: {out}/")
    print(f"\n  Next — run reconstruction:")
    print(f"    python scripts/05_tsdf_fusion.py \\")
    print(f"        --data {args.out} \\")
    print(f"        --mode ssm \\")
    print(f"        --checkpoint {args.checkpoint} \\")
    print(f"        --out outputs/phone_mesh")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
