"""
scripts/05_tsdf_fusion.py
=========================
Fuse SSM-refined (or baseline) depth maps into a TSDF volume,
then extract a coloured triangle mesh via Marching Cubes.

Modes:
  baseline  — per-frame DAV2 depth (scale-aligned to GT)
  ssm       — S5-refined depth (requires trained checkpoint)
  gt        — ground-truth depth (upper-bound reference)

Data sources (--data-type):
  tum     — TUM RGB-D benchmark sequence (has GT depth)
  colmap  — Phone video processed by scripts/00_process_video.py
            (no GT depth; uses metric_scale.txt from COLMAP + DAV2)
  record3d — Record3D / Stray Scanner export (has metric LiDAR depth)

Run on TUM:
    python scripts/05_tsdf_fusion.py \
        --data data/rgbd_dataset_freiburg1_desk \
        --mode ssm \
        --checkpoint outputs/ssm_model/best_model.pt \
        --out outputs/mesh

Run on phone video (COLMAP path):
    python scripts/05_tsdf_fusion.py \
        --data data/phone_scene \
        --data-type colmap \
        --mode ssm \
        --checkpoint outputs/ssm_model_v3/best_model.pt \
        --out outputs/mesh_phone

Run on Record3D / Stray Scanner export:
    python scripts/05_tsdf_fusion.py \
        --data data/record3d_scene \
        --data-type record3d \
        --mode ssm \
        --checkpoint outputs/ssm_model_v3/best_model.pt \
        --out outputs/mesh_record3d
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.tum_dataset import TUMDataset
from src.data.colmap_dataset import ColmapDataset
from src.geometry.scale_align import align_depth_to_gt
from src.geometry.tsdf_fusion import TSDFFusion
from src.models.depth_anything import DepthAnythingV2
from src.models.temporal_depth import DepthRefinementSSM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str,   default="data/rgbd_dataset_freiburg1_desk")
    p.add_argument("--data-type",  type=str,   default="tum",
                   choices=["tum", "colmap", "record3d"],
                   help="Data source type (tum|colmap|record3d)")
    p.add_argument("--mode",       type=str,   default="ssm", choices=["baseline", "ssm", "gt"])
    p.add_argument("--checkpoint", type=str,   default="outputs/ssm_model/best_model.pt")
    p.add_argument("--dav2_size",  type=str,   default="small")
    p.add_argument("--max_frames", type=int,   default=200)
    p.add_argument("--seq_len",    type=int,   default=8)
    p.add_argument("--voxel_size", type=float, default=0.02)
    p.add_argument("--sdf_trunc",  type=float, default=0.08)
    p.add_argument("--out",        type=str,   default="outputs/mesh")
    p.add_argument("--device",     type=str,   default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device

    # Normalise hyphenated arg name (argparse converts - to _)
    data_type = args.data_type

    print(f"\n{'='*60}")
    print(f"  TSDF Fusion → 3D Mesh  [mode={args.mode}, data={data_type}]")
    print(f"{'='*60}\n")

    # ---- Dataset ----
    if data_type == "tum":
        ds = TUMDataset(args.data, max_frames=args.max_frames)
    elif data_type == "colmap":
        ds = ColmapDataset(args.data, max_frames=args.max_frames)
    elif data_type == "record3d":
        # Record3D / Stray Scanner export — same interface as ColmapDataset
        # but may have LiDAR depth. Try to import Record3DDataset; fall back
        # to ColmapDataset (which returns zeros for depth) if not present.
        try:
            from src.data.record3d_dataset import Record3DDataset
            ds = Record3DDataset(args.data, max_frames=args.max_frames)
        except ImportError:
            print("[Warning] Record3DDataset not found — falling back to ColmapDataset")
            ds = ColmapDataset(args.data, max_frames=args.max_frames)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    K = ds[0]["K"].numpy()

    # ---- Determine whether this dataset has metric GT depth ----
    # ColmapDataset returns depth=zeros to signal "no GT".
    # We check the first frame; if all zeros → no-GT path.
    _sample0 = ds[0]
    has_gt_depth = bool((_sample0["depth"] > 0).any().item())

    # ---- For no-GT datasets: load COLMAP-derived metric scale ----
    colmap_metric_scale: float | None = None
    if not has_gt_depth:
        scale_path = Path(args.data) / "metric_scale.txt"
        if scale_path.exists():
            colmap_metric_scale = float(scale_path.read_text().strip())
            print(f"Loaded COLMAP metric scale: {colmap_metric_scale:.4f} m/unit")
        else:
            print("[Warning] No metric_scale.txt found — "
                  "depth will be in DAV2 relative units (scene shape OK, scale wrong).")
            colmap_metric_scale = 1.0   # best effort

    # ---- Models ----
    dav2      = None
    ssm_model = None
    img_h, img_w = 240, 320

    if args.mode in ("baseline", "ssm"):
        dav2 = DepthAnythingV2(size=args.dav2_size, device=device)

    if args.mode == "ssm":
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
        print(f"Loaded SSM: epoch {ckpt['epoch']}, val_AbsRel={ckpt['val_abs_rel']:.4f}")

        # Load DAV2 cache if available
        cache_path = Path(args.checkpoint).parent / f"dav2_depths_{img_h}x{img_w}.npy"
        dav2_cache = np.load(str(cache_path)) if cache_path.exists() else None
        if dav2_cache is not None:
            print(f"Loaded DAV2 cache: {cache_path}")
        else:
            print("No DAV2 cache — computing hints on-the-fly")

    # ---- TSDF fuser ----
    fuser = TSDFFusion(voxel_size=args.voxel_size, sdf_trunc=args.sdf_trunc)

    # ---- Step 1: compute predicted depths ----
    # Two strategies depending on whether the dataset has GT depth:
    #
    # WITH GT DEPTH (TUM / Record3D with LiDAR):
    #   • GT pixels are metric and temporally consistent → use directly.
    #   • Monocular (DAV2/SSM) fills holes, aligned per-frame to GT.
    #
    # WITHOUT GT DEPTH (COLMAP phone video):
    #   • COLMAP-derived metric scale (from scripts/00_process_video.py) applied
    #     globally to DAV2/SSM output. COLMAP poses guarantee cross-frame
    #     consistency; metric scale ensures absolute distances are correct.
    #   • If SSM was trained, it provides temporally smoother depth relative
    #     to DAV2 → less TSDF ghosting from flicker.
    print(f"\nStep 1/2 — Computing depths ({len(ds)} frames, mode={args.mode}, "
          f"has_gt={has_gt_depth})...")

    all_rgbs   = []
    all_depths = []
    all_poses  = []

    # Sliding window buffers for SSM
    hint_buf   = []
    rgb_buf    = []
    ssm_depths = {}   # frame_idx → [H_orig, W_orig] tensor

    def flush_ssm_window(hints, rgbs, indices, ds_ref):
        hint_t = torch.stack(hints).to(device)
        rgb_t  = torch.stack(rgbs).to(device)
        with torch.no_grad():
            out = ssm_model(hint_t, rgb_t, use_parallel=False)
        for t, idx in enumerate(indices):
            sample = ds_ref[idx]
            H_orig = sample["depth"].shape[-2]
            W_orig = sample["depth"].shape[-1]
            refined = F.interpolate(
                out[t:t+1], size=(H_orig, W_orig),
                mode="bilinear", align_corners=False
            ).squeeze().cpu()
            ssm_depths[idx] = refined

    n_gt_only  = 0
    n_hybrid   = 0
    n_mono_scale = 0   # no-GT path: pure monocular × COLMAP scale

    for i, sample in enumerate(tqdm(ds, desc="Depth pass")):
        rgb_t    = sample["rgb"]              # [3, H, W]
        depth_gt = sample["depth"].squeeze()  # [H, W]  metric metres, 0=invalid
        pose     = sample["pose"].numpy()
        H_orig, W_orig = depth_gt.shape
        gt_np   = depth_gt.numpy()
        gt_mask = gt_np > 0

        if args.mode == "gt":
            if has_gt_depth:
                depth_m = gt_np
                n_gt_only += 1
            else:
                print("[Warning] 'gt' mode requested but dataset has no GT depth. "
                      "Switching to baseline.")
                args.mode = "baseline"
                if dav2 is None:
                    dav2 = DepthAnythingV2(size=args.dav2_size, device=device)

        if args.mode in ("baseline", "ssm"):
            # ----------------------------------------------------------------
            # Get raw monocular depth prediction (DAV2 or SSM-refined)
            # ----------------------------------------------------------------
            if args.mode == "ssm":
                if dav2_cache is not None and i < len(dav2_cache):
                    hint = torch.from_numpy(dav2_cache[i]).unsqueeze(0)
                else:
                    with torch.no_grad():
                        d_hint = dav2.predict(
                            rgb_t.unsqueeze(0).to(device),
                            output_size=(img_h, img_w)
                        ).squeeze().cpu()
                    hint = d_hint.unsqueeze(0)

                rgb_r = F.interpolate(
                    rgb_t.unsqueeze(0), size=(img_h, img_w),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
                hint_buf.append(hint)
                rgb_buf.append(rgb_r)

                if len(hint_buf) >= args.seq_len or i == len(ds) - 1:
                    indices = list(range(i - len(hint_buf) + 1, i + 1))
                    flush_ssm_window(hint_buf, rgb_buf, indices, ds)
                    step = max(1, args.seq_len // 2)
                    hint_buf[:] = hint_buf[step:]
                    rgb_buf[:]  = rgb_buf[step:]

                if i in ssm_depths:
                    d_mono = ssm_depths[i].numpy()
                else:
                    with torch.no_grad():
                        d_mono = dav2.predict(
                            rgb_t.unsqueeze(0).to(device),
                            output_size=(H_orig, W_orig)
                        ).squeeze().cpu().numpy()
            else:  # baseline
                with torch.no_grad():
                    d_mono = dav2.predict(
                        rgb_t.unsqueeze(0).to(device),
                        output_size=(H_orig, W_orig)
                    ).squeeze().cpu().numpy()

            # ----------------------------------------------------------------
            # Convert monocular output to metric depth
            # ----------------------------------------------------------------
            if has_gt_depth and gt_mask.sum() > 50:
                # --- Path A: Hybrid GT + monocular hole-filling (TUM / Record3D) ---
                # GT pixels are metric → use directly. Monocular fills holes,
                # aligned per-frame to GT so fill is also metric.
                mono_at_gt = d_mono[gt_mask].clip(1e-4)
                scale = float(np.median(gt_np[gt_mask] / mono_at_gt))

                depth_m = gt_np.copy()              # start with GT
                hole_mask = ~gt_mask
                depth_m[hole_mask] = d_mono[hole_mask] * scale
                n_hybrid += 1

            else:
                # --- Path B: Pure monocular × COLMAP metric scale (phone video) ---
                # COLMAP poses ensure temporal consistency across frames.
                # Global metric scale (computed from COLMAP sparse depths vs DAV2)
                # converts DAV2/SSM relative output to metric metres.
                # No per-frame GT anchor → rely on scale stability of DAV2 + COLMAP.
                scale = colmap_metric_scale if colmap_metric_scale is not None else 1.0
                depth_m = d_mono * scale
                n_mono_scale += 1

        depth_m = np.clip(depth_m, 0.05, 10.0)
        rgb_np  = (rgb_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        all_rgbs.append(rgb_np)
        all_depths.append(depth_m)
        all_poses.append(pose)

    if has_gt_depth:
        gt_coverage = float(np.mean([g > 0 for g in [ds[i]["depth"].squeeze().numpy()
                                                      for i in range(min(10, len(ds)))]]) * 100)
        print(f"  Hybrid fusion: {n_hybrid} frames (GT+fill), {n_gt_only} GT-only")
        print(f"  Approx GT coverage per frame: ~{gt_coverage:.0f}%")
    else:
        print(f"  Pure-mono path: {n_mono_scale} frames × COLMAP scale "
              f"{colmap_metric_scale:.4f}")

    # ---- Step 2: TSDF fusion ----
    print(f"\nStep 2/2 — TSDF integration ({len(all_depths)} frames)...")
    for rgb, depth, pose in tqdm(
        zip(all_rgbs, all_depths, all_poses),
        total=len(all_depths), desc="Fusing"
    ):
        fuser.integrate(rgb, depth, K, pose)

    # ---- Extract mesh ----
    mesh = fuser.extract_mesh()
    print(f"\nRaw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # ---- Post-process: remove floating artifacts ----
    import open3d as o3d

    # 1. Keep only the largest connected component
    #    This removes floating shards from bad frames
    print("Post-processing: removing disconnected components...")
    triangle_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    triangle_clusters  = np.asarray(triangle_clusters)
    cluster_n_tris     = np.asarray(cluster_n_tris)
    largest_cluster    = cluster_n_tris.argmax()
    remove_mask        = triangle_clusters != largest_cluster
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    print(f"  After component filter: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    # 2. Statistical outlier removal on the vertex point cloud
    #    Removes spiky vertices that are statistically far from neighbours
    print("Post-processing: statistical outlier removal...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    mesh = mesh.select_by_index(ind)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    print(f"  After outlier removal:  {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    # 3. Light smoothing to reduce surface noise without losing shape
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.5)
    mesh.compute_vertex_normals()
    print(f"  After smoothing:        {len(mesh.vertices):,} vertices")

    stem = f"scene_{args.mode}"
    fuser.save_mesh(out / f"{stem}.ply", mesh)

    try:
        fuser.save_mesh(out / f"{stem}.glb", mesh)
    except ImportError:
        print("trimesh not installed — skipping .glb export (pip install trimesh)")

    print(f"\nDone. Files in {out}/")
    print(f"  .ply : open in MeshLab / Blender / CloudCompare")
    print(f"  .glb : drag into https://3dviewer.net")


if __name__ == "__main__":
    main()
