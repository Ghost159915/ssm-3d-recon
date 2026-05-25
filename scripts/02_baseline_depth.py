"""
scripts/02_baseline_depth.py
============================
Compute per-frame depth using Depth Anything V2 (NO SSM — baseline).
Scale-align each frame to GT depth using least-squares.
Compute quantitative metrics and save a visualisation.

Run:
    python scripts/02_baseline_depth.py \
        --data data/rgbd_dataset_freiburg1_desk \
        --size small \
        --max_frames 100 \
        --out outputs/baseline
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.data.tum_dataset import TUMDataset
from src.geometry.scale_align import align_depth_to_gt
from src.models.depth_anything import DepthAnythingV2
from src.utils.metrics import MetricsAccumulator
from src.utils.viz import plot_depth_comparison, save_depth_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, default="data/rgbd_dataset_freiburg1_desk")
    p.add_argument("--size",       type=str, default="small", choices=["small", "base", "large"])
    p.add_argument("--max_frames", type=int, default=100)
    p.add_argument("--out",        type=str, default="outputs/baseline")
    p.add_argument("--device",     type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Day 2 — Baseline Depth (Depth Anything V2, no SSM)")
    print(f"{'='*60}\n")

    # Dataset
    ds = TUMDataset(args.data, max_frames=args.max_frames)
    print(f"Loaded {len(ds)} frames.\n")

    # Model
    model = DepthAnythingV2(size=args.size, device=args.device)

    # Metrics
    acc = MetricsAccumulator()

    # Store a few frames for the grid visualisation
    depth_preds_vis = []
    depth_gts_vis   = []
    rgb_vis         = []
    N_VIS           = 8

    for i, sample in enumerate(tqdm(ds, desc="Baseline depth")):
        rgb_t   = sample["rgb"].unsqueeze(0)    # [1, 3, H, W]
        depth_gt = sample["depth"].squeeze()    # [H, W]

        H, W = depth_gt.shape

        with torch.no_grad():
            depth_rel = model.predict(rgb_t, output_size=(H, W)).squeeze()  # [H, W]

        # Scale alignment to GT
        depth_aligned = align_depth_to_gt(depth_rel, depth_gt, mask=depth_gt > 0)

        # Metrics
        acc.update(depth_aligned.unsqueeze(0), depth_gt.unsqueeze(0))

        # Store for visualisation
        if i < N_VIS:
            depth_preds_vis.append(depth_aligned)
            depth_gts_vis.append(depth_gt)
            rgb_vis.append(sample["rgb"])

    # Summary
    results = acc.summary()
    acc.pretty_print("Baseline — Depth Anything V2 (no SSM)")

    # Save metrics to text file
    metrics_path = out / "baseline_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Baseline Depth Metrics (Depth Anything V2)\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Frames : {len(ds)}\n")
        f.write(f"Model  : {args.size}\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Metrics saved → {metrics_path}")

    # Save side-by-side comparison for first frame
    if len(rgb_vis) > 0:
        fig = plot_depth_comparison(
            rgb=rgb_vis[0],
            depth_gt=depth_gts_vis[0],
            depth_pred=depth_preds_vis[0],
            title="Baseline: Depth Anything V2 (frame 0)",
            save_path=str(out / "frame0_comparison.png"),
        )

    # Save grid of predicted depths
    save_depth_grid(
        depths=depth_preds_vis,
        save_path=str(out / "baseline_depth_grid.png"),
        title="Baseline Depth Predictions (first 8 frames)",
    )

    print(f"\nDone. Outputs saved to {out}/")
    print(f"\nExpected baseline AbsRel: ~0.08–0.12 (Depth Anything V2 is strong)")
    print(f"Actual AbsRel: {results['abs_rel']:.4f}")


if __name__ == "__main__":
    main()
