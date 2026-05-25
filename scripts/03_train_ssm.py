"""
scripts/03_train_ssm.py  (v4 — multi-dataset: TUM + ScanNet + ARKitScenes)
=============================================================================
Train the S5 temporal depth REFINEMENT module.

Architecture: CNN(DAV2_depth + RGB) → S5Stack → bounded residual correction
  depth_out = DAV2_hint + Tanh(MLP(ssm_features)) * 0.2

Dataset types supported (auto-detected from folder structure):
  tum         — TUM RGB-D benchmark (rgb.txt / depth.txt)
  scannet     — ScanNet v2 (color/ depth/ pose/ intrinsic/)
  arkitscenes — Apple ARKitScenes (lowres_wide/ lowres_wide.traj)
  record3d    — Record3D / Stray Scanner iPhone LiDAR export
  colmap      — Phone video processed by scripts/00_process_video.py

Run (TUM only — original):
    python scripts/03_train_ssm.py \
        --data data/rgbd_dataset_freiburg1_desk \
               data/rgbd_dataset_freiburg1_xyz \
               data/rgbd_dataset_freiburg1_360 \
        --epochs 60 --out outputs/ssm_model_v3_multi

Run (TUM + ScanNet + ARKitScenes — diverse):
    python scripts/03_train_ssm.py \
        --data data/rgbd_dataset_freiburg1_desk \
               data/rgbd_dataset_freiburg2_desk \
               data/rgbd_dataset_freiburg3_long_office_household \
               data/scannet/scene0000_00 \
               data/scannet/scene0001_00 \
               data/scannet/scene0002_00 \
               data/arkitscenes/Training/4199 \
               data/arkitscenes/Training/4204 \
               data/arkitscenes/Training/4175 \
        --epochs 60 --out outputs/ssm_model_v4_diverse
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.data.dataset_factory import make_dataset, detect_dataset_type
from src.geometry.scale_align import align_depth_to_gt
from src.models.depth_anything import DepthAnythingV2
from src.models.temporal_depth import DepthRefinementSSM, edge_aware_smoothness_loss
from src.utils.metrics import MetricsAccumulator


# ---------------------------------------------------------------------------
# Step 1: Pre-compute and cache DAV2 depth predictions
# ---------------------------------------------------------------------------

def precompute_dav2_depths(
    ds,                          # any dataset with TUMDataset-compatible interface
    dav2: DepthAnythingV2,
    cache_path: Path,
    img_h: int,
    img_w: int,
    device: str,
) -> np.ndarray:
    """
    Run Depth Anything V2 on all frames and cache to disk.
    Saves a float32 array of shape [N, H, W] with scale-aligned depths.
    Works with TUM, ScanNet, ARKitScenes, or any dataset returning
    {'rgb': [3,H,W], 'depth': [1,H,W]} dicts.
    """
    if cache_path.exists():
        print(f"[Step 1] Loading cached DAV2 depths from {cache_path}")
        return np.load(str(cache_path))

    print(f"[Step 1] Pre-computing DAV2 depths for {len(ds)} frames...")
    depths = []

    for i, sample in enumerate(tqdm(ds, desc="DAV2 inference")):
        rgb_t    = sample["rgb"].unsqueeze(0)   # [1, 3, H_orig, W_orig]
        depth_gt = sample["depth"].squeeze()    # [H_orig, W_orig]

        with torch.no_grad():
            d_rel = dav2.predict(rgb_t, output_size=(img_h, img_w)).squeeze().cpu()
            depth_gt_resized = F.interpolate(
                depth_gt.unsqueeze(0).unsqueeze(0).float(),
                size=(img_h, img_w), mode="nearest"
            ).squeeze()

        # Scale-align to GT where GT is available (TUM/ScanNet/ARKitScenes all have it)
        mask = depth_gt_resized > 0
        if mask.sum() > 50:
            d_aligned = align_depth_to_gt(d_rel, depth_gt_resized, mask=mask)
        else:
            d_aligned = d_rel   # COLMAP/no-GT path — keep relative

        d_max  = d_aligned.max()
        d_norm = (d_aligned / (d_max + 1e-8)).clamp(0, 1)
        depths.append(d_norm.numpy().astype(np.float32))

    arr = np.stack(depths, axis=0)
    np.save(str(cache_path), arr)
    print(f"[Step 1] Saved to {cache_path}  shape={arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Step 2: Sequence dataset for training
# ---------------------------------------------------------------------------

class DepthSequenceDataset(Dataset):
    """
    Yields windows of (depth_hints, rgb, depth_gt) for S5 training.

    depth_hints : [T, 1, H, W]  pre-computed DAV2 depth (normalised)
    rgb         : [T, 3, H, W]  RGB frames
    depth_gt    : [T, H, W]     GT metric depth (for supervision)
    gt_norm     : float         normalisation factor to bring gt into [0,1]
    """

    def __init__(
        self,
        tum_ds: TUMDataset,
        dav2_depths: np.ndarray,   # [N, H, W]
        seq_len: int = 8,
        stride: int = 4,
        img_h: int = 240,
        img_w: int = 320,
    ):
        self.ds          = tum_ds
        self.dav2_depths = dav2_depths
        self.seq_len     = seq_len
        self.img_h       = img_h
        self.img_w       = img_w
        self.starts      = list(range(0, max(1, len(tum_ds) - seq_len + 1), stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end   = min(start + self.seq_len, len(self.ds))

        rgbs       = []
        depth_hints= []
        depth_gts  = []

        for i in range(start, end):
            s = self.ds[i]
            rgb_t  = s["rgb"]                    # [3, H, W]
            gt_t   = s["depth"].squeeze()        # [H, W]
            hint_t = torch.from_numpy(self.dav2_depths[i])  # [H, W]

            # Resize if needed
            if rgb_t.shape[-2:] != (self.img_h, self.img_w):
                rgb_t = F.interpolate(rgb_t.unsqueeze(0),
                                      size=(self.img_h, self.img_w),
                                      mode="bilinear", align_corners=False).squeeze(0)
                gt_t  = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0),
                                      size=(self.img_h, self.img_w),
                                      mode="nearest").squeeze()
                hint_t = F.interpolate(hint_t.unsqueeze(0).unsqueeze(0),
                                       size=(self.img_h, self.img_w),
                                       mode="bilinear", align_corners=False).squeeze()

            rgbs.append(rgb_t)
            depth_hints.append(hint_t.unsqueeze(0))   # [1, H, W]
            depth_gts.append(gt_t)

        # Pad short sequences
        while len(rgbs) < self.seq_len:
            rgbs.append(rgbs[-1])
            depth_hints.append(depth_hints[-1])
            depth_gts.append(depth_gts[-1])

        return {
            "rgb":        torch.stack(rgbs),         # [T, 3, H, W]
            "depth_hint": torch.stack(depth_hints),  # [T, 1, H, W]
            "depth_gt":   torch.stack(depth_gts),    # [T, H, W]
        }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",          type=str,   nargs="+",
                   default=["data/rgbd_dataset_freiburg1_desk"],
                   help="One or more dataset directories (TUM/ScanNet/ARKitScenes/COLMAP — auto-detected)")
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--seq_len",       type=int,   default=16)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--lambda_smooth", type=float, default=0.05)
    p.add_argument("--d_model",       type=int,   default=128)
    p.add_argument("--d_state",       type=int,   default=64)
    p.add_argument("--n_layers",      type=int,   default=3)
    p.add_argument("--img_h",         type=int,   default=240)
    p.add_argument("--img_w",         type=int,   default=320)
    p.add_argument("--dav2_size",     type=str,   default="small")
    p.add_argument("--val_frac",      type=float, default=0.1)
    p.add_argument("--out",           type=str,   default="outputs/ssm_model_v3")
    p.add_argument("--device",        type=str,   default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    print(f"  Train S5 Depth Refinement  (v3 — multi-sequence)")
    print(f"{'='*60}")
    print(f"  device={device}  d_model={args.d_model}  d_state={args.d_state}")
    print(f"  n_layers={args.n_layers}  seq_len={args.seq_len}  epochs={args.epochs}")
    print(f"  sequences ({len(args.data)}): {[Path(d).name for d in args.data]}\n")

    # ---- Step 1: Pre-compute DAV2 depths per sequence ----
    dav2 = DepthAnythingV2(size=args.dav2_size, device=device)

    all_seq_datasets = []   # list of (dataset, np.ndarray) per sequence

    for seq_path in args.data:
        ds_type  = detect_dataset_type(seq_path)
        # ScanNet and ARKitScenes record at 25-30 fps; stride to ~5-10 fps
        stride   = 5 if ds_type == "scannet" else (3 if ds_type == "arkitscenes" else 1)
        seq_ds   = make_dataset(seq_path, dataset_type=ds_type, stride=stride)
        seq_name = Path(seq_path).name
        # Per-sequence cache — reused if already computed
        cache_path = out / f"dav2_depths_{seq_name}_{args.img_h}x{args.img_w}.npy"
        dav2_depths = precompute_dav2_depths(
            seq_ds, dav2, cache_path, args.img_h, args.img_w, device
        )
        all_seq_datasets.append((seq_ds, dav2_depths))

    # Also keep the primary sequence cache under the old name for eval script compatibility
    primary_cache = out / f"dav2_depths_{args.img_h}x{args.img_w}.npy"
    if not primary_cache.exists():
        import shutil
        first_seq_name = Path(args.data[0]).name
        first_cache = out / f"dav2_depths_{first_seq_name}_{args.img_h}x{args.img_w}.npy"
        if first_cache.exists():
            shutil.copy(first_cache, primary_cache)

    # Free DAV2 from GPU memory
    del dav2
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    total_frames = sum(len(ds) for ds, _ in all_seq_datasets)
    print(f"\nTotal frames across all sequences: {total_frames}")

    # ---- Step 2: Build per-sequence window datasets, then combine ----
    # Windows must NOT cross sequence boundaries — build each separately
    from torch.utils.data import ConcatDataset

    seq_window_datasets = []
    for raw_ds, dav2_depths in all_seq_datasets:
        seq_ds = DepthSequenceDataset(
            raw_ds, dav2_depths,
            seq_len=args.seq_len, stride=4,
            img_h=args.img_h, img_w=args.img_w,
        )
        seq_window_datasets.append(seq_ds)
        name = Path(raw_ds.root).name if hasattr(raw_ds, "root") else type(raw_ds).__name__
        print(f"  {name}: {len(raw_ds)} frames → {len(seq_ds)} windows")

    full_ds = ConcatDataset(seq_window_datasets)

    n_val   = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"\nWindows: {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=0)

    # ---- Step 3: Model ----
    model = DepthRefinementSSM(
        img_size=(args.img_h, args.img_w),
        cnn_channels=32,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    history = {"train_loss": [], "val_abs_rel": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []

        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            hint = batch["depth_hint"].to(device)   # [B, T, 1, H, W]
            rgb  = batch["rgb"].to(device)           # [B, T, 3, H, W]
            gt   = batch["depth_gt"].to(device)      # [B, T, H, W]

            B, T, _, H, W = hint.shape

            # Forward: process each sequence
            # Reshape for model: [B, T, 1, H, W] and [B, T, 3, H, W]
            pred_list = []
            for b in range(B):
                refined = model(hint[b], rgb[b], use_parallel=True)  # [T, 1, H, W]
                pred_list.append(refined)
            pred = torch.stack(pred_list)   # [B, T, 1, H, W]

            # Normalise GT to [0,1] for loss (match the hint normalisation)
            gt_max = gt.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)  # [B, T, 1, 1]
            gt_norm = (gt / gt_max).clamp(0, 1)               # [B, T, H, W]
            gt_norm_c = gt_norm.unsqueeze(2)                   # [B, T, 1, H, W]

            valid = (gt > 0).unsqueeze(2)

            if valid.sum() > 0:
                l1 = (pred[valid] - gt_norm_c[valid]).abs().mean()
            else:
                l1 = pred.sum() * 0.0

            # Smoothness on flattened frames
            pred_flat = pred.view(B * T, 1, H, W)
            rgb_flat  = rgb.view(B * T, 3, H, W)
            smooth = edge_aware_smoothness_loss(pred_flat, rgb_flat)

            loss = l1 + args.lambda_smooth * smooth

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            train_losses.append(loss.item())

        scheduler.step()
        mean_train = sum(train_losses) / max(len(train_losses), 1)
        history["train_loss"].append(mean_train)

        # --- Validate ---
        model.eval()
        acc = MetricsAccumulator()

        with torch.no_grad():
            for batch in val_loader:
                hint = batch["depth_hint"].to(device)
                rgb  = batch["rgb"].to(device)
                gt   = batch["depth_gt"]     # keep on CPU for metrics

                B, T, _, H, W = hint.shape
                for b in range(B):
                    refined = model(hint[b], rgb[b], use_parallel=False)  # [T, 1, H, W]
                    refined_cpu = refined.cpu()

                    # Align refined (normalised [0,1]) back to metric scale
                    # using the same scale factor from the hint
                    # We use the GT to re-align for fair evaluation
                    for t in range(T):
                        d_pred_t = refined_cpu[t, 0]   # [H, W]
                        d_gt_t   = gt[b, t]             # [H, W]
                        mask_t   = d_gt_t > 0

                        if mask_t.sum() < 10:
                            continue

                        # Scale the normalised prediction to metric
                        from src.geometry.scale_align import align_depth_to_gt
                        d_metric = align_depth_to_gt(d_pred_t, d_gt_t, mask=mask_t)
                        acc.update(d_metric.unsqueeze(0), d_gt_t.unsqueeze(0))

        val_metrics = acc.summary()
        val_abs_rel = val_metrics.get("abs_rel", float("inf"))
        history["val_abs_rel"].append(val_abs_rel)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={mean_train:.4f}  "
              f"val_AbsRel={val_abs_rel:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_abs_rel < best_val:
            best_val = val_abs_rel
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_abs_rel": val_abs_rel,
                "args": vars(args),
                "model_class": "DepthRefinementSSM",
            }, out / "best_model.pt")
            print(f"  → Saved best model (AbsRel={val_abs_rel:.4f})")

    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best val AbsRel : {best_val:.4f}")
    print(f"  Sequences trained: {[Path(d).name for d in args.data]}")
    print(f"  Checkpoint → {out}/best_model.pt")
    print(f"  Run evaluation:")
    print(f"    python scripts/04_evaluate.py --checkpoint {out}/best_model.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
