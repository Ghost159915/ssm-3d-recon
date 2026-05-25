"""
src/geometry/tsdf_fusion.py
===========================
TSDF (Truncated Signed Distance Function) volumetric fusion using Open3D.

Takes a sequence of metric depth maps with camera poses and fuses them
into a single watertight triangle mesh via Marching Cubes.

Why TSDF over NeRF:
  - Deterministic, no training required
  - Runs in seconds (not hours)
  - Open3D ScalableTSDFVolume handles large scenes adaptively
  - Clean watertight mesh output

Usage
-----
    from src.geometry.tsdf_fusion import TSDFFusion

    fuser = TSDFFusion(voxel_size=0.02, sdf_trunc=0.08)

    # Add frames one by one
    for rgb, depth, pose in frames:
        fuser.integrate(rgb, depth, K, pose)

    # Extract mesh
    mesh = fuser.extract_mesh()
    fuser.save_mesh("output/scene.ply")
    fuser.save_mesh("output/scene.glb")   # web-compatible

    # Or extract point cloud
    pcd = fuser.extract_pointcloud()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def _require_open3d():
    if not OPEN3D_AVAILABLE:
        raise ImportError(
            "open3d >= 0.18 is required for TSDF fusion.\n"
            "Install: pip install open3d"
        )


# ---------------------------------------------------------------------------
# TSDF Fusion
# ---------------------------------------------------------------------------

class TSDFFusion:
    """
    Incremental TSDF fusion using Open3D ScalableTSDFVolume.

    Parameters
    ----------
    voxel_size : float   size of each voxel in metres (0.02 = 2 cm recommended)
    sdf_trunc  : float   TSDF truncation distance in metres.
                         Rule of thumb: 4 × voxel_size
    color_type : str     "rgb" or "none"
    """

    def __init__(
        self,
        voxel_size: float = 0.02,
        sdf_trunc: float = 0.08,
        color_type: str = "rgb",
    ):
        _require_open3d()
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc

        color = (
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if color_type == "rgb"
            else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=color,
        )

        self.frame_count = 0

    # ------------------------------------------------------------------
    def integrate(
        self,
        rgb: Union[np.ndarray, torch.Tensor],
        depth: Union[np.ndarray, torch.Tensor],
        K: Union[np.ndarray, torch.Tensor],
        pose_c2w: Union[np.ndarray, torch.Tensor],
        depth_scale: float = 1000.0,
        max_depth: float = 5.0,
    ) -> None:
        """
        Integrate one RGBD frame into the TSDF volume.

        Parameters
        ----------
        rgb       : [H, W, 3] uint8 or float32 [0,1]
        depth     : [H, W]   float32 in metres
        K         : [3, 3]   camera intrinsic matrix
        pose_c2w  : [4, 4]   camera-to-world SE(3)
        depth_scale: float   Open3D expects depth in millimetres internally;
                             we convert metres → mm with this factor (1000.0).
        max_depth : float    clip depth beyond this (metres) before fusion.
        """
        # Convert tensors to numpy
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if isinstance(K, torch.Tensor):
            K = K.cpu().numpy()
        if isinstance(pose_c2w, torch.Tensor):
            pose_c2w = pose_c2w.cpu().numpy()

        # Ensure float64 for Open3D
        K = K.astype(np.float64)
        pose_c2w = pose_c2w.astype(np.float64)

        # RGB: ensure uint8 [H, W, 3]
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        if rgb.shape[0] == 3:   # [3, H, W] → [H, W, 3]
            rgb = rgb.transpose(1, 2, 0)

        # Depth: clip invalid, convert to uint16 (mm) for Open3D
        depth_m = depth.squeeze().astype(np.float32)
        depth_m = np.clip(depth_m, 0.0, max_depth)

        # Open3D Image from arrays
        o3d_rgb   = o3d.geometry.Image(rgb)
        o3d_depth = o3d.geometry.Image(
            (depth_m * depth_scale).astype(np.uint16)
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb,
            o3d_depth,
            depth_scale=depth_scale,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )

        h, w = depth_m.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h,
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
        )

        # Open3D expects extrinsic = world-to-camera (inverse of pose_c2w)
        extrinsic = np.linalg.inv(pose_c2w)

        self.volume.integrate(rgbd, intrinsic, extrinsic)
        self.frame_count += 1

    # ------------------------------------------------------------------
    def integrate_sequence(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        depths: Union[np.ndarray, torch.Tensor],
        K: Union[np.ndarray, torch.Tensor],
        poses_c2w: Union[np.ndarray, torch.Tensor],
        verbose: bool = True,
    ) -> None:
        """
        Integrate a full sequence of frames.

        Parameters
        ----------
        rgbs      : [T, H, W, 3] uint8 or [T, 3, H, W] float32
        depths    : [T, H, W] or [T, 1, H, W]   float32, metres
        K         : [3, 3]   shared intrinsics
        poses_c2w : [T, 4, 4]  camera-to-world poses
        """
        if isinstance(rgbs, torch.Tensor):
            rgbs = rgbs.cpu().numpy()
        if isinstance(depths, torch.Tensor):
            depths = depths.cpu().numpy()
        if isinstance(K, torch.Tensor):
            K = K.cpu().numpy()
        if isinstance(poses_c2w, torch.Tensor):
            poses_c2w = poses_c2w.cpu().numpy()

        T = len(rgbs)
        from tqdm import tqdm
        iter_ = tqdm(range(T), desc="TSDF fusion") if verbose else range(T)
        for i in iter_:
            rgb_i   = rgbs[i]
            depth_i = depths[i].squeeze()
            pose_i  = poses_c2w[i]
            self.integrate(rgb_i, depth_i, K, pose_i)

    # ------------------------------------------------------------------
    def extract_mesh(self) -> "o3d.geometry.TriangleMesh":
        """
        Run Marching Cubes on the TSDF volume and return the mesh.
        """
        _require_open3d()
        print(f"[TSDFFusion] Extracting mesh from {self.frame_count} frames...")
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    # ------------------------------------------------------------------
    def extract_pointcloud(self) -> "o3d.geometry.PointCloud":
        """Extract a dense point cloud from the TSDF volume."""
        _require_open3d()
        return self.volume.extract_point_cloud()

    # ------------------------------------------------------------------
    def save_mesh(
        self,
        path: Union[str, Path],
        mesh: Optional["o3d.geometry.TriangleMesh"] = None,
    ) -> None:
        """
        Save the extracted mesh to file.
        Supports: .ply, .obj, .glb (web-compatible via trimesh).

        Parameters
        ----------
        path : str | Path    output file path
        mesh : optional pre-extracted mesh (avoids re-running Marching Cubes)
        """
        _require_open3d()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if mesh is None:
            mesh = self.extract_mesh()

        suffix = path.suffix.lower()
        if suffix in (".ply", ".obj"):
            o3d.io.write_triangle_mesh(str(path), mesh)
        elif suffix == ".glb":
            # Export via trimesh for web-compatible glTF
            try:
                import trimesh
                vertices = np.asarray(mesh.vertices)
                faces    = np.asarray(mesh.triangles)
                colors   = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
                tm = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=(colors * 255).astype(np.uint8) if colors is not None else None,
                )
                tm.export(str(path))
            except ImportError:
                raise ImportError("trimesh is required for .glb export: pip install trimesh")
        else:
            o3d.io.write_triangle_mesh(str(path), mesh)

        print(f"[TSDFFusion] Saved mesh → {path}")

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the TSDF volume and start fresh."""
        self.volume.reset()
        self.frame_count = 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not OPEN3D_AVAILABLE:
        print("open3d not installed — skipping TSDF smoke test")
    else:
        import tempfile

        print("TSDF fusion smoke test")
        fuser = TSDFFusion(voxel_size=0.05, sdf_trunc=0.2)

        H, W = 240, 320
        K = np.array([[320, 0, 160], [0, 320, 120], [0, 0, 1]], dtype=np.float64)

        # Simulate 5 frames with forward-moving camera
        for i in range(5):
            rgb = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
            # Flat depth at ~1 metre with noise
            depth = np.ones((H, W), dtype=np.float32) + np.random.rand(H, W) * 0.05
            pose = np.eye(4)
            pose[2, 3] = i * 0.05   # move forward 5 cm per frame
            fuser.integrate(rgb, depth, K, pose)

        mesh = fuser.extract_mesh()
        print(f"Mesh vertices : {len(mesh.vertices)}")
        print(f"Mesh triangles: {len(mesh.triangles)}")

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            fuser.save_mesh(f.name, mesh)
            print(f"Saved: {f.name}")

        print("TSDF smoke test passed.")
