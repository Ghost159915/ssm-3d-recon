"""
src/semantics/lift_labels.py
============================
Semantic labeling: 2D detections → 3D mesh labels.

Pipeline:
  1. GroundingDINO: open-vocabulary detection on keyframes
     ("chair . table . floor . wall . monitor . keyboard")
  2. SAM2: generate fine-grained segmentation masks from detected boxes
  3. Back-projection: for each labeled pixel (u, v), compute 3D point
     using depth map + camera pose
  4. Voting: assign the most frequent label to each nearby mesh vertex

Usage
-----
    from src.semantics.lift_labels import SemanticLifter

    lifter = SemanticLifter(
        text_prompt="chair . table . floor . wall",
        device="mps",
    )
    labeled_mesh = lifter.lift(
        mesh, frames, depths, poses, K, keyframe_step=10
    )
    lifter.save_labeled_mesh(labeled_mesh, "output/scene_semantic.ply")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helper: check optional dependencies
# ---------------------------------------------------------------------------

def _try_import_grounding_dino():
    try:
        from groundingdino.util.inference import load_model, predict
        return load_model, predict
    except ImportError:
        return None, None


def _try_import_sam2():
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return build_sam2, SAM2ImagePredictor
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# Colour palette for semantic classes
# ---------------------------------------------------------------------------

LABEL_COLOURS = [
    [255,  99,  71],   # tomato     — chair
    [ 70, 130, 180],   # steel blue — table
    [144, 238, 144],   # light green— floor
    [210, 180, 140],   # tan        — wall
    [255, 215,   0],   # gold       — monitor
    [147, 112, 219],   # medium purple — keyboard
    [255, 165,   0],   # orange     — other
    [100, 149, 237],   # cornflower — background
]


def label_to_color(label_idx: int) -> np.ndarray:
    """Return [R, G, B] uint8 for a given label index."""
    return np.array(LABEL_COLOURS[label_idx % len(LABEL_COLOURS)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Semantic Lifter
# ---------------------------------------------------------------------------

class SemanticLifter:
    """
    Lift 2D open-vocabulary detections to 3D mesh vertex labels.

    Parameters
    ----------
    text_prompt : str   GroundingDINO prompt, dot-separated labels.
                        e.g. "chair . table . floor . wall"
    box_threshold: float  GroundingDINO box confidence threshold
    text_threshold: float GroundingDINO text confidence threshold
    device      : str   "cpu" | "mps" | "cuda"
    """

    def __init__(
        self,
        text_prompt: str = "chair . table . floor . wall . monitor . keyboard",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cpu",
    ):
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

        # Parse label names from prompt
        self.label_names = [l.strip() for l in text_prompt.split(".") if l.strip()]

        # Try to load models (optional deps)
        self._gd_load_model, self._gd_predict = _try_import_grounding_dino()
        self._sam2_build, self._sam2_predictor = _try_import_sam2()

        self._gd_model = None
        self._sam2_predictor_inst = None

        if self._gd_load_model is None:
            warnings.warn(
                "GroundingDINO not installed. Semantic lifting will use a dummy "
                "random-colour fallback.\n"
                "Install: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
            )
        else:
            print("[SemanticLifter] GroundingDINO available.")

        if self._sam2_build is None:
            warnings.warn(
                "SAM2 not installed. Using bounding-box masks instead of precise segments.\n"
                "Install: pip install git+https://github.com/facebookresearch/sam2.git"
            )
        else:
            print("[SemanticLifter] SAM2 available.")

    # ------------------------------------------------------------------
    def _detect_frame(
        self,
        rgb_np: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Run GroundingDINO on a single frame.

        Parameters
        ----------
        rgb_np : [H, W, 3] uint8

        Returns
        -------
        boxes  : [M, 4] xyxy normalised
        labels : list of M label strings
        """
        if self._gd_load_model is None or self._gd_model is None:
            # Fallback: no detections
            return np.zeros((0, 4)), []

        from PIL import Image as PILImage
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((800,)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_pil = PILImage.fromarray(rgb_np)
        img_t = transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            boxes, logits, phrases = self._gd_predict(
                model=self._gd_model,
                image=img_t,
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

        boxes_np = boxes.cpu().numpy()   # normalised [M, 4]
        return boxes_np, phrases

    # ------------------------------------------------------------------
    def _segment_frame(
        self,
        rgb_np: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        """
        Run SAM2 on boxes to get per-instance masks.

        Parameters
        ----------
        rgb_np    : [H, W, 3] uint8
        boxes_xyxy: [M, 4] pixel-space xyxy boxes

        Returns
        -------
        masks : [M, H, W] bool
        """
        H, W = rgb_np.shape[:2]
        if len(boxes_xyxy) == 0:
            return np.zeros((0, H, W), dtype=bool)

        if self._sam2_predictor_inst is None:
            # Fallback: use bounding-box rectangles as masks
            masks = np.zeros((len(boxes_xyxy), H, W), dtype=bool)
            for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
                masks[i, max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
            return masks

        self._sam2_predictor_inst.set_image(rgb_np)
        masks, _, _ = self._sam2_predictor_inst.predict(
            box=boxes_xyxy,
            multimask_output=False,
        )
        return masks[:, 0, :, :]   # [M, H, W]

    # ------------------------------------------------------------------
    def _back_project_labels(
        self,
        label_map: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        pose_c2w: np.ndarray,
        valid_depth_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Back-project labeled pixels to 3D points.

        Parameters
        ----------
        label_map : [H, W] int  label index per pixel (-1 = unlabeled)
        depth     : [H, W] float32 metres
        K         : [3, 3]
        pose_c2w  : [4, 4]

        Returns
        -------
        pts_3d : [N, 3] float32  world-space 3D points
        labels : [N]    int      label indices
        """
        H, W = depth.shape[:2]

        # Valid pixels: labeled + positive depth
        mask = (label_map >= 0) & (depth > 0) & (depth < 10.0)
        if valid_depth_mask is not None:
            mask &= valid_depth_mask

        v_coords, u_coords = np.where(mask)   # row=v, col=u
        if len(u_coords) == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=int)

        # Unproject: pixel (u, v) + depth z → camera-frame 3D
        z = depth[v_coords, u_coords]
        x = (u_coords - K[0, 2]) * z / K[0, 0]
        y = (v_coords - K[1, 2]) * z / K[1, 1]
        pts_cam = np.stack([x, y, z], axis=1)   # [N, 3]

        # Camera → world
        R = pose_c2w[:3, :3]
        t = pose_c2w[:3, 3]
        pts_world = (R @ pts_cam.T).T + t        # [N, 3]

        labels = label_map[v_coords, u_coords]   # [N]
        return pts_world.astype(np.float32), labels.astype(int)

    # ------------------------------------------------------------------
    def lift(
        self,
        mesh,
        rgb_frames: List[np.ndarray],
        depth_frames: List[np.ndarray],
        poses_c2w: List[np.ndarray],
        K: np.ndarray,
        keyframe_step: int = 10,
        n_neighbors: int = 3,
    ):
        """
        Lift 2D labels to 3D mesh vertex labels.

        Parameters
        ----------
        mesh         : open3d.geometry.TriangleMesh
        rgb_frames   : list of [H, W, 3] uint8
        depth_frames : list of [H, W] float32 metres
        poses_c2w    : list of [4, 4] float64
        K            : [3, 3] intrinsics
        keyframe_step: int  only process every N-th frame for speed
        n_neighbors  : int  voting neighbourhood (unused currently — we use
                            KD-tree nearest vertex)

        Returns
        -------
        mesh with vertex_colors set according to semantic labels
        """
        try:
            import open3d as o3d
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("open3d and scipy are required for label lifting.")

        vertices = np.asarray(mesh.vertices)   # [V, 3]
        V = len(vertices)

        # Vote accumulator: [V, n_labels] counts
        n_labels = len(self.label_names)
        vote_counts = np.zeros((V, n_labels), dtype=np.int32)

        kd_tree = cKDTree(vertices)

        # Process keyframes
        keyframe_indices = list(range(0, len(rgb_frames), keyframe_step))
        print(f"[SemanticLifter] Processing {len(keyframe_indices)} keyframes "
              f"(step={keyframe_step}) out of {len(rgb_frames)} total.")

        for i in keyframe_indices:
            rgb   = rgb_frames[i]
            depth = depth_frames[i].squeeze()
            pose  = poses_c2w[i]

            H, W = rgb.shape[:2]

            # Detect
            boxes_norm, phrases = self._detect_frame(rgb)

            if len(boxes_norm) == 0:
                continue

            # Denormalise boxes
            boxes_px = boxes_norm.copy()
            boxes_px[:, [0, 2]] *= W
            boxes_px[:, [1, 3]] *= H
            boxes_px = boxes_px.astype(int)

            # Segment
            masks = self._segment_frame(rgb, boxes_px)   # [M, H, W]

            # Build per-pixel label map
            label_map = np.full((H, W), -1, dtype=int)
            for m_idx, (phrase, mask) in enumerate(zip(phrases, masks)):
                # Find which label index this phrase maps to
                label_idx = self._phrase_to_label_idx(phrase)
                label_map[mask] = label_idx

            # Back-project
            pts_3d, pt_labels = self._back_project_labels(label_map, depth, K, pose)

            if len(pts_3d) == 0:
                continue

            # Assign to nearest mesh vertex
            dists, vtx_indices = kd_tree.query(pts_3d, k=1, workers=-1)
            valid = dists < 0.1   # only assign if within 10 cm

            for j in range(len(pt_labels)):
                if valid[j] and 0 <= pt_labels[j] < n_labels:
                    vote_counts[vtx_indices[j], pt_labels[j]] += 1

        # Assign winning label to each vertex
        vertex_labels = vote_counts.argmax(axis=1)   # [V]
        # Vertices with no votes → -1 (background colour)
        no_vote = vote_counts.sum(axis=1) == 0
        vertex_labels[no_vote] = n_labels   # "background" index

        # Assign colours
        all_colors = LABEL_COLOURS + [[128, 128, 128]]   # last = background grey
        vertex_colors = np.array(
            [all_colors[l % len(all_colors)] for l in vertex_labels],
            dtype=np.float64
        ) / 255.0

        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        print(f"[SemanticLifter] Labeled {int((~no_vote).sum())} / {V} vertices.")
        return mesh

    # ------------------------------------------------------------------
    def _phrase_to_label_idx(self, phrase: str) -> int:
        """Map a detected phrase to the closest label index."""
        phrase_lower = phrase.lower()
        for i, name in enumerate(self.label_names):
            if name.lower() in phrase_lower or phrase_lower in name.lower():
                return i
        return len(self.label_names) - 1   # fallback: last label

    # ------------------------------------------------------------------
    def save_labeled_mesh(self, mesh, path: str) -> None:
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d is required.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(path), mesh)
        print(f"[SemanticLifter] Saved labeled mesh → {path}")

    # ------------------------------------------------------------------
    def print_legend(self) -> None:
        print("\nSemantic label colour legend:")
        for i, name in enumerate(self.label_names):
            c = LABEL_COLOURS[i % len(LABEL_COLOURS)]
            print(f"  [{i}] {name:<20} RGB{tuple(c)}")
        print(f"  [-] background        RGB(128,128,128)\n")
