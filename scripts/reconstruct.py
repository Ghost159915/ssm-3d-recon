"""
scripts/reconstruct.py
======================
Full pipeline entry point — runs the complete SSM-3DRecon pipeline:

    data → Depth Anything V2 (baseline) → S5 SSM (temporal) →
    TSDF fusion → 3D mesh → semantic labels

Usage:
    python scripts/reconstruct.py \
        --data data/rgbd_dataset_freiburg1_desk \
        --checkpoint outputs/ssm_model/best_model.pt \
        --query "chair . table . floor . wall . monitor . keyboard" \
        --out outputs/final

Or, skip training and use GT depth (sanity check):
    python scripts/reconstruct.py \
        --data data/rgbd_dataset_freiburg1_desk \
        --depth_mode gt \
        --out outputs/gt_mesh
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(
        description="SSM-3DRecon: video → 3D scene reconstruction with S5 SSM"
    )
    p.add_argument("--data",        type=str,   required=True,
                   help="Path to TUM RGB-D dataset folder")
    p.add_argument("--checkpoint",  type=str,   default=None,
                   help="Trained SSM checkpoint (.pt). If None, uses baseline only.")
    p.add_argument("--depth_mode",  type=str,   default="ssm",
                   choices=["gt", "baseline", "ssm"],
                   help="Depth source for TSDF fusion")
    p.add_argument("--query",       type=str,
                   default="chair . table . floor . wall . monitor . keyboard",
                   help="GroundingDINO text query for semantic labeling")
    p.add_argument("--max_frames",  type=int,   default=300)
    p.add_argument("--voxel_size",  type=float, default=0.02)
    p.add_argument("--out",         type=str,   default="outputs/final")
    p.add_argument("--device",      type=str,   default="auto")
    p.add_argument("--skip_train",  action="store_true",
                   help="Skip SSM training (requires --checkpoint to exist)")
    p.add_argument("--skip_semantic", action="store_true",
                   help="Skip semantic labeling step")
    return p.parse_args()


def run(cmd: list, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=True)
    return result.returncode


def main():
    args = parse_args()
    out = Path(args.out)
    ssm_out = out / "ssm_model"
    mesh_out = out / "mesh"
    sem_out  = out / "semantic"
    eval_out = out / "eval"

    scripts = Path(__file__).parent

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           SSM-3DRecon: Full Pipeline                        ║
║   S5 State Space Model for Temporal Depth Consistency       ║
╚══════════════════════════════════════════════════════════════╝

Dataset   : {args.data}
Mode      : {args.depth_mode}
Output    : {args.out}
Device    : {args.device}
""")

    # Step 1: Baseline depth evaluation
    run([
        sys.executable, str(scripts / "02_baseline_depth.py"),
        "--data", args.data,
        "--max_frames", str(min(100, args.max_frames)),
        "--out", str(out / "baseline"),
        "--device", args.device,
    ], "Step 1 — Baseline depth (Depth Anything V2)")

    # Step 2: Train S5 (unless skipped / checkpoint provided)
    if args.depth_mode == "ssm" and not args.skip_train:
        checkpoint = args.checkpoint or str(ssm_out / "best_model.pt")
        if not Path(checkpoint).exists():
            run([
                sys.executable, str(scripts / "03_train_ssm.py"),
                "--data", args.data,
                "--out", str(ssm_out),
                "--device", args.device,
            ], "Step 2 — Train S5 temporal depth consistency")
        else:
            print(f"\nCheckpoint found at {checkpoint} — skipping training.")
    elif args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None

    # Step 3: Evaluate (if SSM trained)
    if checkpoint and Path(checkpoint).exists():
        run([
            sys.executable, str(scripts / "04_evaluate.py"),
            "--data", args.data,
            "--checkpoint", checkpoint,
            "--max_frames", str(min(200, args.max_frames)),
            "--out", str(eval_out),
            "--device", args.device,
        ], "Step 3 — Quantitative evaluation (baseline vs SSM)")

    # Step 4: TSDF fusion
    tsdf_cmd = [
        sys.executable, str(scripts / "05_tsdf_fusion.py"),
        "--data", args.data,
        "--mode", args.depth_mode if checkpoint else "baseline",
        "--max_frames", str(args.max_frames),
        "--voxel_size", str(args.voxel_size),
        "--out", str(mesh_out),
        "--device", args.device,
    ]
    if checkpoint:
        tsdf_cmd += ["--checkpoint", checkpoint]

    run(tsdf_cmd, "Step 4 — TSDF fusion → 3D mesh")

    # Step 5: Semantic labeling
    mesh_path = mesh_out / f"scene_{args.depth_mode if checkpoint else 'baseline'}.ply"
    if not args.skip_semantic and mesh_path.exists():
        run([
            sys.executable, str(scripts / "06_semantic_label.py"),
            "--data", args.data,
            "--mesh", str(mesh_path),
            "--query", args.query,
            "--out", str(sem_out),
        ], "Step 5 — Semantic label lifting (GroundingDINO + SAM2)")
    elif not mesh_path.exists():
        print(f"\n[Warning] Mesh not found at {mesh_path} — skipping semantics.")

    # Summary
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Pipeline complete!                                         ║
╠══════════════════════════════════════════════════════════════╣
║  Outputs:                                                   ║
║    Baseline metrics  : {str(out / 'baseline'):<34} ║
║    Evaluation        : {str(eval_out):<34} ║
║    3D Mesh (.ply)    : {str(mesh_out):<34} ║
║    3D Mesh (.glb)    : {str(mesh_out):<34} ║
║    Semantic mesh     : {str(sem_out):<34} ║
╚══════════════════════════════════════════════════════════════╝

View .glb in browser: drag into https://3dviewer.net
""")


if __name__ == "__main__":
    main()
