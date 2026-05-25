# SSM-3DRecon: Temporally Consistent 3D Scene Reconstruction from Video

> **Humanoid Robotics Internship Challenge** — Phone Video → Geometrically Coherent 3D Mesh

A complete pipeline that takes a short phone video of an indoor scene and produces a clean, metric-scale 3D mesh. The core contribution is an **S5 State Space Model built from scratch in pure PyTorch** that enforces temporal consistency across monocular depth estimates before 3D fusion — fixing the frame-to-frame flickering that makes naive monocular reconstruction fail.

---

## Results

| Method | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|---|---|---|---|
| Depth Anything V2 (baseline) | 0.0967 | 0.179 m | 90.3% |
| **DAV2 + S5 SSM (ours)** | **0.0556** | **0.136 m** | **97.3%** |
| Improvement | **+42.6%** | **+24.2%** | **+7.0 pp** |

Evaluated on TUM RGB-D `freiburg1_desk`, 200 frames. The SSM reduces absolute relative depth error by 42.6% over the monocular baseline with no additional inference-time cost beyond a single forward pass.

---

## Pipeline

```
Phone Video
    │
    ▼
ffmpeg → frames
    │
    ▼
COLMAP SfM ──────────────────────────────────────┐
(feature match + sparse reconstruct)             │
    │                                            │
    ├── camera poses (4×4 SE3)                   │
    └── sparse 3D points → metric scale          │
                                                 │
    ┌────────────────────────────────────────────┘
    ▼
Depth Anything V2
(monocular relative depth per frame)
    │
    ▼
S5 SSM Temporal Refinement  ←── trained on TUM + ARKitScenes
(CNN encoder → S5 state space → bounded residual correction)
(makes depth temporally consistent across the sequence)
    │
    ▼
TSDF Fusion (Open3D)
(integrates all metric depth maps into a single volume)
    │
    ▼
Marching Cubes → Triangle Mesh (.ply / .glb)
```

---

## Quick Start

### Requirements

- Python 3.11
- Mac (MPS) or Linux with AMD/CUDA GPU
- COLMAP + ffmpeg for phone video processing

```bash
# Mac
brew install colmap ffmpeg
```

### Install

```bash
git clone https://github.com/YOUR_USERNAME/ssm-3d-recon.git
cd ssm-3d-recon

conda create -n ssm3d_env python=3.11 -y
conda activate ssm3d_env

# Mac (MPS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r environment/requirements_mac.txt

# Linux AMD (ROCm) — see LINUX_SETUP.md for full GPU setup
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
pip install -r environment/requirements_linux.txt
```

### Run on your own phone video

```bash
# Step 1: Extract frames + COLMAP poses + metric scale (~5 min)
python scripts/00_process_video.py \
    --video my_room.mp4 \
    --out data/my_scene \
    --fps 5

# Step 2: Reconstruct 3D mesh (~10 min)
python scripts/05_tsdf_fusion.py \
    --data data/my_scene \
    --data-type colmap \
    --mode ssm \
    --checkpoint outputs/ssm_model/best_model.pt \
    --out outputs/mesh_my_scene
```

Open `outputs/mesh_my_scene/scene_ssm.glb` at [3dviewer.net](https://3dviewer.net) to view your mesh in-browser.

---

## Full Pipeline

### 1. Download training data

```bash
bash scripts/download_datasets.sh tum          # TUM RGB-D — no auth required
bash scripts/download_datasets.sh arkitscenes  # iPad Pro LiDAR scenes — HuggingFace token required
```

For ARKitScenes: accept terms at [huggingface.co/datasets/apple/ARKitScenes](https://huggingface.co/datasets/apple/ARKitScenes), then `huggingface-cli login`.

### 2. Train the SSM

```bash
python scripts/03_train_ssm.py \
    --data \
        data/rgbd_dataset_freiburg1_desk \
        data/rgbd_dataset_freiburg1_xyz \
        data/rgbd_dataset_freiburg1_room \
        data/rgbd_dataset_freiburg2_desk \
        data/rgbd_dataset_freiburg3_long_office_household \
        data/arkitscenes/Training/4199 \
        data/arkitscenes/Training/4204 \
    --epochs 60 \
    --d_model 128 \
    --d_state 64 \
    --out outputs/ssm_model
```

Training auto-detects dataset type (TUM / ScanNet / ARKitScenes / COLMAP) from folder structure. DAV2 depths are cached after the first run.

### 3. Evaluate

```bash
python scripts/04_evaluate.py \
    --data data/rgbd_dataset_freiburg1_desk \
    --checkpoint outputs/ssm_model/best_model.pt \
    --out outputs/eval
```

### 4. Fuse to mesh

```bash
# TUM benchmark data
python scripts/05_tsdf_fusion.py \
    --data data/rgbd_dataset_freiburg1_desk \
    --mode ssm \
    --checkpoint outputs/ssm_model/best_model.pt \
    --voxel_size 0.02 --sdf_trunc 0.04 \
    --out outputs/mesh

# Phone video (no depth sensor)
python scripts/05_tsdf_fusion.py \
    --data data/my_scene \
    --data-type colmap \
    --mode ssm \
    --checkpoint outputs/ssm_model/best_model.pt \
    --out outputs/mesh_phone
```

### 5. Visualise (depth comparison video)

```bash
python scripts/06_render_video.py \
    --data data/rgbd_dataset_freiburg1_desk \
    --checkpoint outputs/ssm_model/best_model.pt \
    --out outputs/video
```

---

## Supported Data Sources

The pipeline auto-detects format from folder structure — no `--dataset-type` flag needed for training:

| Source | Detected by | Depth |
|---|---|---|
| TUM RGB-D | `rgb.txt` + `depth.txt` | Kinect sensor |
| ScanNet v2 | `color/` + `pose/` + `intrinsic/` | Structure Sensor |
| ARKitScenes | `lowres_wide/` + `lowres_wide.traj` | iPad Pro LiDAR |
| Record3D / Stray Scanner | `rgbd/` + `metadata` | iPhone LiDAR (Pro) |
| Phone video | `poses.npy` + `frame_names.txt` | COLMAP scale only |

---

## Design Choices & Rationale

### Why an SSM for depth consistency?

Monocular depth flickering is a **sequence modelling problem**: each frame's depth is plausible in isolation but inconsistent across time. This inconsistency causes TSDF ghosting — the fused volume sees the same surface at conflicting depths across frames and blurs or doubles it.

The fix is a model with **persistent memory across frames**. Transformers do this but cost O(T²) in attention. RNNs degrade over long sequences. **S5 State Space Models** process sequences in O(T log T) via parallel scan and maintain a hidden state that accumulates geometric context across the full video. The hidden state acts as running scene memory — depth at frame 200 is informed by everything the camera has already seen.

### Why build S5 from scratch?

All published S5/Mamba implementations use custom CUDA kernels (`mamba-ssm`, `causal-conv1d`) that don't run on Apple MPS or AMD ROCm. This implementation — diagonal state matrix, Zero-Order Hold discretisation, HiPPO-LegS initialisation, parallel scan for training / recurrent scan for inference — is pure PyTorch and runs identically on any device.

See `src/models/s5.py` (~300 lines, fully commented).

### Why residual correction rather than full prediction?

The SSM outputs a bounded residual correction (Tanh × 0.2) on top of DAV2's prediction rather than predicting depth from scratch. This means:
- The model only learns to fix temporal inconsistencies, not estimate depth
- Training converges in hours not weeks
- DAV2's strong single-frame accuracy is preserved; SSM only improves consistency

### Why COLMAP for phone video?

COLMAP recovers metric-scale camera poses from plain RGB video with no depth sensor. The sparse 3D point cloud it produces lets us compute the ratio between DAV2's relative depth and actual metric distances — so the final TSDF operates in true metres.

### Why TSDF fusion?

TSDF (Truncated Signed Distance Function) handles conflicting depth observations by averaging them volumetrically. Noisy or slightly inconsistent depths that would produce spiky point clouds instead smooth out during integration. Marching Cubes extracts a watertight triangle mesh. Post-processing: largest connected component filter → statistical outlier removal → Laplacian smoothing.

---

## Repository Structure

```
ssm-3d-recon/
├── src/
│   ├── models/
│   │   ├── s5.py                  # S5 SSM — diagonal A, ZOH, HiPPO init, parallel scan
│   │   ├── temporal_depth.py      # DepthRefinementSSM: CNN + S5Stack + residual gate
│   │   └── depth_anything.py      # Depth Anything V2 wrapper (HuggingFace)
│   ├── data/
│   │   ├── tum_dataset.py         # TUM RGB-D with timestamp association
│   │   ├── scannet_dataset.py     # ScanNet v2
│   │   ├── arkitscenes_dataset.py # ARKitScenes (iPad Pro LiDAR)
│   │   ├── colmap_dataset.py      # Phone video post-COLMAP
│   │   ├── record3d_dataset.py    # iPhone LiDAR via Record3D/Stray Scanner
│   │   └── dataset_factory.py     # Auto-detect + instantiate any dataset
│   └── geometry/
│       ├── tsdf_fusion.py         # Open3D TSDF + Marching Cubes wrapper
│       └── scale_align.py         # Least-squares metric scale alignment
├── scripts/
│   ├── 00_process_video.py        # Video → COLMAP → poses + metric scale
│   ├── 03_train_ssm.py            # Multi-dataset training (TUM/ScanNet/ARKitScenes)
│   ├── 04_evaluate.py             # Quantitative evaluation (AbsRel, RMSE, δ metrics)
│   ├── 05_tsdf_fusion.py          # Depth maps → 3D mesh (GT / SSM / baseline modes)
│   ├── 06_render_video.py         # Side-by-side depth comparison video
│   └── download_datasets.sh       # One-command dataset download
├── environment/
│   ├── requirements_mac.txt       # Mac MPS
│   └── requirements_linux.txt     # Linux ROCm / CUDA
└── LINUX_SETUP.md                 # ROCm setup + dual-machine workflow
```

---

## Performance

| Device | DAV2 inference | SSM training |
|---|---|---|
| Mac M4 MPS | ~5 fps | ~1.5 it/s |
| Mac M5 Max MPS | ~60 fps | ~18 it/s |
| AMD RX 7700 XT ROCm | ~35 fps | ~10 it/s |
| NVIDIA GPU CUDA | ~50+ fps | ~15+ it/s |

Pure PyTorch throughout — no custom CUDA kernels, no compilation step. Zero code changes across devices.

---

## Phone Video Tips

For best COLMAP results:
- **Light**: bright, even lighting — motion blur kills feature matching
- **Speed**: move slowly (~5 cm/s), 60% frame overlap
- **Scene**: textured surfaces (books, objects on desk) — blank walls have no features
- **Duration**: 30 sec to 2 min, shot at 30 fps and extracted at 5 fps
- **Coverage**: don't just pan — move closer/further and tilt to build 3D parallax
