"""
scripts/06_semantic_label.py
============================
Add semantic labels to the reconstructed mesh using GroundingDINO + SAM2.

Run:
    python scripts/06_semantic_label.py \
        --data data/rgbd_dataset_freiburg1_desk \
        --mesh outputs/mesh/scene_ssm.ply \
        --query "chair . table . floor . wall . monitor . keyboard" \
        --out outputs/semantic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data.tum_dataset import TUMDataset
from src.semantics.lift_labels import SemanticLifter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",  type=str, default="data/rgbd_dataset_freiburg1_desk")
    p.add_argument("--mesh",  type=str, default="outputs/mesh/scene_ssm.ply")
    p.add_argument("--query", type=str, default="chair . table . floor . wall . monitor . keyboard")
    p.add_argument("--keyframe_step", type=int, default=10)
    p.add_argument("--max_frames",    type=int, default=300)
    p.add_argument("--out",   type=str, default="outputs/semantic")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Day 6 — Semantic Label Lifting")
    print(f"  Query: '{args.query}'")
    print(f"{'='*60}\n")

    # Load mesh
    try:
        import open3d as o3d
    except ImportError:
        print("ERROR: open3d required. pip install open3d")
        return

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # Load dataset frames
    ds = TUMDataset(args.data, max_frames=args.max_frames)

    rgb_frames   = []
    depth_frames = []
    poses        = []

    for sample in ds:
        rgb_np = (sample["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rgb_frames.append(rgb_np)
        depth_frames.append(sample["depth"].squeeze().numpy())
        poses.append(sample["pose"].numpy())

    K = ds[0]["K"].numpy()

    # Semantic lifter
    lifter = SemanticLifter(
        text_prompt=args.query,
        device=args.device,
    )
    lifter.print_legend()

    labeled_mesh = lifter.lift(
        mesh=mesh,
        rgb_frames=rgb_frames,
        depth_frames=depth_frames,
        poses_c2w=poses,
        K=K,
        keyframe_step=args.keyframe_step,
    )

    # Save
    stem = Path(args.mesh).stem
    lifter.save_labeled_mesh(labeled_mesh, str(out / f"{stem}_semantic.ply"))

    # Also save as .glb for web viewer
    try:
        from src.geometry.tsdf_fusion import TSDFFusion
        fuser = TSDFFusion()  # just for the save_mesh helper
        fuser.save_mesh(out / f"{stem}_semantic.glb", labeled_mesh)
    except Exception as e:
        print(f"[Warning] .glb export failed: {e}")

    print(f"\nDone. Semantic mesh saved to {out}/")
    print(f"Open {stem}_semantic.ply in MeshLab or Blender to inspect labels.")


if __name__ == "__main__":
    main()
