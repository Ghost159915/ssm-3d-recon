"""
scripts/06_render_video.py
==========================
Render two videos:
  1. depth_comparison.mp4  — side-by-side: RGB | GT | Baseline | SSM | Error
  2. mesh_turntable.mp4    — rotating Open3D mesh render

Run:
    python scripts/06_render_video.py \
        --data data/rgbd_dataset_freiburg1_desk \
        --checkpoint outputs/ssm_model/best_model.pt \
        --mesh outputs/mesh/scene_ssm.ply \
        --out outputs/video \
        --max_frames 200
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from src.data.tum_dataset import TUMDataset
from src.geometry.scale_align import align_depth_to_gt
from src.models.depth_anything import DepthAnythingV2
from src.models.temporal_depth import DepthRefinementSSM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, default="data/rgbd_dataset_freiburg1_desk")
    p.add_argument("--checkpoint", type=str, default="outputs/ssm_model/best_model.pt")
    p.add_argument("--mesh",       type=str, default="outputs/mesh/scene_ssm.ply")
    p.add_argument("--out",        type=str, default="outputs/video")
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--fps",        type=int, default=10)
    p.add_argument("--seq_len",    type=int, default=8)
    p.add_argument("--dav2_size",  type=str, default="small")
    p.add_argument("--device",     type=str, default="auto")
    p.add_argument("--skip_mesh",  action="store_true", help="Skip turntable render")
    p.add_argument("--skip_depth", action="store_true", help="Skip depth video")
    return p.parse_args()


def colorise_depth(d: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Float depth [H,W] → RGB uint8 [H,W,3] using viridis colormap."""
    if vmin is None: vmin = np.percentile(d[d > 0], 2) if (d > 0).any() else 0
    if vmax is None: vmax = np.percentile(d[d > 0], 98) if (d > 0).any() else 1
    d_norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    rgba = cm.viridis(d_norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Video 1: Depth comparison
# ---------------------------------------------------------------------------

def render_depth_video(args, out: Path):
    """Produce depth_comparison.mp4 showing RGB | GT | Baseline | SSM | Error."""
    try:
        import cv2
    except ImportError:
        print("opencv-python required: pip install opencv-python")
        return

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device

    # Load models
    dav2 = DepthAnythingV2(size=args.dav2_size, device=device)

    ckpt  = torch.load(args.checkpoint, map_location=device)
    margs = ckpt["args"]
    img_h = margs.get("img_h", 240)
    img_w = margs.get("img_w", 320)

    ssm_model = DepthRefinementSSM(
        img_size=(img_h, img_w),
        cnn_channels=32,
        d_model=margs.get("d_model", 64),
        d_state=margs.get("d_state", 32),
        n_layers=margs.get("n_layers", 3),
    ).to(device)
    ssm_model.load_state_dict(ckpt["model_state"])
    ssm_model.eval()

    cache_path = Path(args.checkpoint).parent / f"dav2_depths_{img_h}x{img_w}.npy"
    dav2_cache = np.load(str(cache_path)) if cache_path.exists() else None

    ds = TUMDataset(args.data, max_frames=args.max_frames)

    # Collect all frames
    hint_buf = []
    rgb_buf  = []
    ssm_out  = {}   # frame_idx → [H, W] refined at img resolution

    def flush(hints, rgbs, indices):
        ht = torch.stack(hints).to(device)
        rt = torch.stack(rgbs).to(device)
        with torch.no_grad():
            out = ssm_model(ht, rt, use_parallel=False)
        for t, idx in enumerate(indices):
            ssm_out[idx] = out[t, 0].cpu().numpy()

    print("Collecting SSM outputs...")
    for i, sample in enumerate(tqdm(ds, desc="SSM pass")):
        rgb_t    = sample["rgb"]
        if dav2_cache is not None and i < len(dav2_cache):
            hint = torch.from_numpy(dav2_cache[i]).unsqueeze(0)
        else:
            with torch.no_grad():
                hint = dav2.predict(
                    rgb_t.unsqueeze(0).to(device), output_size=(img_h, img_w)
                ).squeeze().cpu().unsqueeze(0)
        rgb_r = F.interpolate(
            rgb_t.unsqueeze(0), size=(img_h, img_w),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        hint_buf.append(hint)
        rgb_buf.append(rgb_r)
        if len(hint_buf) >= args.seq_len or i == len(ds) - 1:
            flush(hint_buf, rgb_buf, list(range(i - len(hint_buf) + 1, i + 1)))
            step = max(1, args.seq_len // 2)
            hint_buf[:] = hint_buf[step:]
            rgb_buf[:] = rgb_buf[step:]

    # Determine video size from first frame
    sample0  = ds[0]
    H, W = sample0["depth"].squeeze().shape
    panel_w  = W
    panel_h  = H
    n_panels = 5
    frame_w  = panel_w * n_panels
    frame_h  = panel_h + 40   # +40 for title bar

    video_path = str(out / "depth_comparison.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, args.fps, (frame_w, frame_h))

    print(f"\nRendering depth_comparison.mp4 ({len(ds)} frames @ {args.fps} fps)...")
    for i, sample in enumerate(tqdm(ds, desc="Rendering")):
        rgb_t    = sample["rgb"]               # [3, H, W]
        depth_gt = sample["depth"].squeeze()   # [H, W]
        H_orig, W_orig = depth_gt.shape

        # Baseline
        with torch.no_grad():
            d_rel = dav2.predict(
                rgb_t.unsqueeze(0).to(device), output_size=(H_orig, W_orig)
            ).squeeze().cpu()
        d_base = align_depth_to_gt(d_rel, depth_gt, mask=depth_gt > 0).numpy()

        # SSM refined (upsample from img res)
        if i in ssm_out:
            ref_small = ssm_out[i]
            ref_full  = F.interpolate(
                torch.from_numpy(ref_small).unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig), mode="bilinear", align_corners=False
            ).squeeze().numpy()
            d_ssm = align_depth_to_gt(
                torch.from_numpy(ref_full), depth_gt, mask=depth_gt > 0
            ).numpy()
        else:
            d_ssm = d_base.copy()

        gt_np = depth_gt.numpy()
        vmin  = np.percentile(gt_np[gt_np > 0], 2)
        vmax  = np.percentile(gt_np[gt_np > 0], 98)

        # Error map
        mask_np = gt_np > 0
        err = np.zeros_like(gt_np)
        err[mask_np] = np.abs(d_ssm[mask_np] - gt_np[mask_np])
        err_vmax = np.percentile(err[mask_np], 95) if mask_np.any() else 1.0

        rgb_np = (rgb_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_rgb    = colorise_depth(gt_np, vmin, vmax)
        base_rgb  = colorise_depth(d_base, vmin, vmax)
        ssm_rgb   = colorise_depth(d_ssm, vmin, vmax)
        err_rgb   = colorise_depth(err, 0, err_vmax)

        panels = [rgb_np, gt_rgb, base_rgb, ssm_rgb, err_rgb]
        row    = np.concatenate(panels, axis=1)   # [H, 5W, 3]

        # Title bar
        canvas = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 30
        canvas[40:, :, :] = row
        labels = ["RGB", "GT Depth", "DAV2 Baseline", "S5 SSM Refined", "Abs Error"]
        for j, lbl in enumerate(labels):
            x = j * panel_w + panel_w // 2 - len(lbl) * 4
            cv2.putText(canvas, lbl, (x, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(canvas, f"Frame {i:04d}", (frame_w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        # OpenCV expects BGR
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved → {video_path}")


# ---------------------------------------------------------------------------
# Video 2: Mesh turntable
# ---------------------------------------------------------------------------

def render_turntable(mesh_path: str, out: Path, fps: int = 20, n_frames: int = 120):
    """Render a 360° turntable of the PLY mesh using Open3D offscreen rendering."""
    try:
        import open3d as o3d
        import cv2
    except ImportError:
        print("open3d + opencv-python required for turntable render")
        return

    print(f"\nRendering mesh_turntable.mp4 ({n_frames} frames)...")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Centre the mesh
    bbox   = mesh.get_axis_aligned_bounding_box()
    centre = bbox.get_center()
    mesh.translate(-centre)

    # Try offscreen renderer (Open3D ≥ 0.16)
    try:
        W_vid, H_vid = 960, 540
        renderer = o3d.visualization.rendering.OffscreenRenderer(W_vid, H_vid)
        renderer.scene.set_background([0.12, 0.12, 0.12, 1.0])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", mesh, mat)

        # Camera orbit
        extent = np.linalg.norm(bbox.get_extent())
        cam_dist = extent * 1.5

        video_path = str(out / "mesh_turntable.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (W_vid, H_vid))

        for k in tqdm(range(n_frames), desc="Turntable"):
            angle = 2 * np.pi * k / n_frames
            cam_x = cam_dist * np.sin(angle)
            cam_z = cam_dist * np.cos(angle)
            cam_y = extent * 0.3

            renderer.setup_camera(
                60.0,
                [cam_x, cam_y, cam_z],   # eye
                [0.0, 0.0, 0.0],          # look-at (origin)
                [0.0, 1.0, 0.0],          # up
            )
            img = np.asarray(renderer.render_to_image())
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"Saved → {video_path}")

    except Exception as e:
        print(f"Offscreen renderer failed ({e})")
        print("Falling back: saving 8 static view screenshots instead...")
        _save_static_views(mesh, out)


def _save_static_views(mesh, out: Path, n_views: int = 8):
    """Fallback: save N static view images of the mesh."""
    import open3d as o3d

    extent  = np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent())
    cam_dist = extent * 1.5

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=960, height=540)
    vis.add_geometry(mesh)
    ctr = vis.get_view_control()

    for k in range(n_views):
        angle_deg = 360 * k / n_views
        ctr.rotate(360 / n_views * 10, 0)
        vis.poll_events()
        vis.update_renderer()
        img_path = str(out / f"mesh_view_{k:02d}.png")
        vis.capture_screen_image(img_path)
        print(f"  Saved view {k} → {img_path}")

    vis.destroy_window()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_depth:
        render_depth_video(args, out)

    if not args.skip_mesh:
        render_turntable(args.mesh, out, fps=args.fps, n_frames=120)

    print(f"\nAll videos saved to {out}/")
    print("  depth_comparison.mp4  — frame-by-frame depth comparison")
    print("  mesh_turntable.mp4    — 360° rotating mesh")


if __name__ == "__main__":
    main()
