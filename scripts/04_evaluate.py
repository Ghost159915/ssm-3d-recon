"""
scripts/04_evaluate.py
======================
Quantitative evaluation: Baseline DAV2 vs S5-refined depth.

Loads the trained DepthRefinementSSM checkpoint and evaluates both
the baseline (per-frame DAV2) and SSM-refined depth against GT.

Run:
    python scripts/04_evaluate.py \
        --data data/rgbd_dataset_freiburg1_desk \
        --checkpoint outputs/ssm_model/best_model.pt \
        --out outputs/evaluation
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.tum_dataset import TUMDataset
from src.geometry.scale_align import align_depth_to_gt
from src.models.depth_anything import DepthAnythingV2
from src.models.temporal_depth import DepthRefinementSSM
from src.utils.metrics import MetricsAccumulator
from src.utils.viz import plot_depth_comparison, print_metrics_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        type=str, default="data/rgbd_dataset_freiburg1_desk")
    p.add_argument("--checkpoint",  type=str, default="outputs/ssm_model/best_model.pt")
    p.add_argument("--dav2_size",   type=str, default="small")
    p.add_argument("--max_frames",  type=int, default=200)
    p.add_argument("--seq_len",     type=int, default=8)
    p.add_argument("--out",         type=str, default="outputs/evaluation")
    p.add_argument("--device",      type=str, default="auto")
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

    print(f"\n{'='*60}")
    print(f"  Quantitative Evaluation: Baseline DAV2 vs S5 SSM")
    print(f"{'='*60}\n")

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    margs = ckpt["args"]
    img_h = margs.get("img_h", 240)
    img_w = margs.get("img_w", 320)

    print(f"Checkpoint: epoch {ckpt['epoch']}, "
          f"train val_AbsRel={ckpt['val_abs_rel']:.4f}")
    print(f"Model size: {img_h}x{img_w}, "
          f"d_model={margs.get('d_model',64)}, d_state={margs.get('d_state',32)}\n")

    # ---- Load DAV2 (baseline) ----
    dav2 = DepthAnythingV2(size=args.dav2_size, device=device)

    # ---- Load trained SSM ----
    ssm_model = DepthRefinementSSM(
        img_size=(img_h, img_w),
        cnn_channels=32,
        d_model=margs.get("d_model", 64),
        d_state=margs.get("d_state", 32),
        n_layers=margs.get("n_layers", 3),
    ).to(device)
    ssm_model.load_state_dict(ckpt["model_state"])
    ssm_model.eval()

    # ---- Load pre-computed DAV2 hints (from training cache) ----
    cache_path = Path(args.checkpoint).parent / f"dav2_depths_{img_h}x{img_w}.npy"
    if cache_path.exists():
        print(f"Loading cached DAV2 hints: {cache_path}")
        dav2_cache = np.load(str(cache_path))   # [N, H, W]
    else:
        dav2_cache = None
        print("No DAV2 cache found — will compute hints on-the-fly")

    # ---- Dataset ----
    ds = TUMDataset(args.data, max_frames=args.max_frames)

    acc_baseline = MetricsAccumulator()
    acc_ssm      = MetricsAccumulator()

    # Sliding window buffer for SSM
    hint_buffer = []
    rgb_buffer  = []
    gt_buffer   = []
    idx_buffer  = []

    def evaluate_window(hints, rgbs, gts, frame_indices):
        """Run SSM on a window and accumulate metrics for all frames."""
        hint_t = torch.stack(hints).to(device)   # [T, 1, H, W]
        rgb_t  = torch.stack(rgbs).to(device)    # [T, 3, H, W]

        with torch.no_grad():
            refined = ssm_model(hint_t, rgb_t, use_parallel=False)  # [T, 1, H, W]

        for t, (d_gt, fi) in enumerate(zip(gts, frame_indices)):
            d_ref = refined[t, 0].cpu()   # [H, W]
            mask  = d_gt > 0
            if mask.sum() < 10:
                continue
            d_metric = align_depth_to_gt(d_ref, d_gt, mask=mask)
            acc_ssm.update(d_metric.unsqueeze(0), d_gt.unsqueeze(0))

    print(f"Evaluating {len(ds)} frames...\n")

    for i, sample in enumerate(tqdm(ds, desc="Evaluating")):
        rgb_t    = sample["rgb"]            # [3, H_orig, W_orig]
        depth_gt = sample["depth"].squeeze() # [H_orig, W_orig]
        H_orig, W_orig = depth_gt.shape

        # ---- Baseline: DAV2 per-frame ----
        with torch.no_grad():
            d_rel = dav2.predict(
                rgb_t.unsqueeze(0).to(device),
                output_size=(H_orig, W_orig)
            ).squeeze().cpu()
        d_base = align_depth_to_gt(d_rel, depth_gt, mask=depth_gt > 0)
        acc_baseline.update(d_base.unsqueeze(0), depth_gt.unsqueeze(0))

        # ---- Prepare SSM hint ----
        if dav2_cache is not None and i < len(dav2_cache):
            hint_np = dav2_cache[i]   # [H, W] at training resolution
            hint = torch.from_numpy(hint_np).unsqueeze(0)   # [1, H, W]
        else:
            # Compute hint on-the-fly
            with torch.no_grad():
                d_hint = dav2.predict(
                    rgb_t.unsqueeze(0).to(device),
                    output_size=(img_h, img_w)
                ).squeeze().cpu()
            hint = d_hint.unsqueeze(0)   # [1, H, W]

        # Resize GT for SSM evaluation
        gt_resized = F.interpolate(
            depth_gt.unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w), mode="nearest"
        ).squeeze()

        # Resize RGB for SSM
        rgb_resized = F.interpolate(
            rgb_t.unsqueeze(0),
            size=(img_h, img_w), mode="bilinear", align_corners=False
        ).squeeze(0)

        hint_buffer.append(hint)
        rgb_buffer.append(rgb_resized)
        gt_buffer.append(gt_resized)
        idx_buffer.append(i)

        # Evaluate when buffer is full (or last frame)
        if len(hint_buffer) >= args.seq_len or i == len(ds) - 1:
            evaluate_window(hint_buffer, rgb_buffer, gt_buffer, idx_buffer)
            # Slide by half-window
            step = max(1, args.seq_len // 2)
            hint_buffer  = hint_buffer[step:]
            rgb_buffer   = rgb_buffer[step:]
            gt_buffer    = gt_buffer[step:]
            idx_buffer   = idx_buffer[step:]

    # ---- Results ----
    baseline_results = acc_baseline.summary()
    ssm_results      = acc_ssm.summary()

    print_metrics_table({
        "DAV2 Baseline (no SSM)": baseline_results,
        "DAV2 + S5 SSM":          ssm_results,
    }, title="Depth Evaluation Results")

    improvement = (
        (baseline_results["abs_rel"] - ssm_results["abs_rel"])
        / baseline_results["abs_rel"] * 100
    )
    print(f"AbsRel:  {baseline_results['abs_rel']:.4f} → {ssm_results['abs_rel']:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    print(f"RMSE:    {baseline_results['rmse']:.4f} → {ssm_results['rmse']:.4f} m")
    print(f"δ<1.25:  {baseline_results['delta_1']*100:.2f}% → {ssm_results['delta_1']*100:.2f}%\n")

    # Save
    results = {
        "baseline": baseline_results,
        "ssm":      ssm_results,
        "abs_rel_improvement_pct": improvement,
        "frames_evaluated": len(ds),
    }
    with open(out / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out}/eval_results.json")

    # Save comparison plot for first frame
    sample0   = ds[0]
    rgb0      = sample0["rgb"]
    depth_gt0 = sample0["depth"].squeeze()
    H0, W0    = depth_gt0.shape

    with torch.no_grad():
        d_rel0  = dav2.predict(rgb0.unsqueeze(0).to(device), output_size=(H0, W0)).squeeze().cpu()
        d_base0 = align_depth_to_gt(d_rel0, depth_gt0, mask=depth_gt0 > 0)

        if dav2_cache is not None:
            hint0 = torch.from_numpy(dav2_cache[0]).unsqueeze(0)
        else:
            hint0 = F.interpolate(d_rel0.unsqueeze(0).unsqueeze(0),
                                  size=(img_h, img_w), mode="bilinear",
                                  align_corners=False).squeeze()
            hint0 = hint0.unsqueeze(0)

        rgb0_r = F.interpolate(rgb0.unsqueeze(0), size=(img_h, img_w),
                               mode="bilinear", align_corners=False).squeeze(0)
        refined0 = ssm_model(
            hint0.unsqueeze(0).to(device),  # [1, 1, H, W] (single frame, no temporal)
            rgb0_r.unsqueeze(0).to(device),
            use_parallel=False
        )[0, 0].cpu()

        # Upsample refined to original GT resolution before alignment
        refined0_full = F.interpolate(
            refined0.unsqueeze(0).unsqueeze(0),
            size=(H0, W0), mode="bilinear", align_corners=False
        ).squeeze()
        d_ref0 = align_depth_to_gt(refined0_full, depth_gt0, mask=depth_gt0 > 0)

    plot_depth_comparison(
        rgb=rgb0,
        depth_gt=depth_gt0,
        depth_pred=d_base0,
        depth_refined=d_ref0,
        title=f"Baseline vs S5-refined  (AbsRel: {baseline_results['abs_rel']:.4f} → {ssm_results['abs_rel']:.4f})",
        save_path=str(out / "frame0_comparison.png"),
    )
    print(f"Comparison plot → {out}/frame0_comparison.png")


if __name__ == "__main__":
    main()
