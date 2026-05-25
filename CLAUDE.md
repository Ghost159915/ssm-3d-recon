# Project Handoff — ssm-3d-recon
> Read this at the start of every new Claude session (Mac or Linux).
> It describes the full project state, what works, what's next.

---

## What This Project Is

Humanoid Robotics internship challenge: take a short phone video of an indoor room and produce a geometrically coherent 3D mesh. Submission is a public GitHub repo with README, example outputs, and design notes.

**Approach**: phone video → COLMAP poses → Depth Anything V2 monocular depth → S5 SSM temporal refinement → TSDF fusion → triangle mesh.

The S5 SSM is the core contribution — built from scratch in pure PyTorch (no CUDA extensions), it makes depth temporally consistent across frames. Evaluated result: **+42.6% AbsRel improvement** over DAV2 baseline (0.0967 → 0.0556).

---

## Repository Layout

```
ssm-3d-recon/
├── src/
│   ├── models/
│   │   ├── s5.py                  # S5 SSM from scratch — diagonal A, ZOH, HiPPO init, parallel scan
│   │   ├── temporal_depth.py      # DepthRefinementSSM: CNN encoder + S5Stack + Tanh×0.2 residual
│   │   └── depth_anything.py      # Depth Anything V2 wrapper (HuggingFace, size=small/base/large)
│   ├── data/
│   │   ├── tum_dataset.py         # TUM RGB-D (rgb.txt/depth.txt/groundtruth.txt)
│   │   ├── scannet_dataset.py     # ScanNet v2 (color/ depth/ pose/ intrinsic/)
│   │   ├── arkitscenes_dataset.py # ARKitScenes (lowres_wide/ lowres_wide.traj)
│   │   ├── colmap_dataset.py      # Phone video output (poses.npy / intrinsics.npy / frame_names.txt)
│   │   ├── record3d_dataset.py    # iPhone LiDAR via Record3D or Stray Scanner
│   │   └── dataset_factory.py     # Auto-detects format from folder structure — use make_dataset()
│   └── geometry/
│       ├── tsdf_fusion.py         # Open3D TSDFFusion wrapper + Marching Cubes + mesh export
│       └── scale_align.py         # Least-squares scale alignment (s*pred ≈ gt)
├── scripts/
│   ├── 00_process_video.py        # Phone video → ffmpeg frames → COLMAP → poses.npy + metric_scale.txt
│   ├── 03_train_ssm.py            # Multi-dataset SSM training (v4 — auto-detects TUM/ScanNet/ARKit)
│   ├── 04_evaluate.py             # AbsRel / RMSE / δ<1.25 evaluation with visualisation
│   ├── 05_tsdf_fusion.py          # Depth → 3D mesh (modes: gt / baseline / ssm; data-types: tum/colmap/record3d)
│   ├── 06_render_video.py         # Side-by-side depth comparison video (depth_comparison.mp4)
│   └── download_datasets.sh       # bash scripts/download_datasets.sh [tum|arkitscenes|all]
├── environment/
│   ├── requirements_mac.txt       # Mac MPS
│   └── requirements_linux.txt     # Linux ROCm / CUDA
├── README.md                      # Public submission README (clean, well written)
├── LINUX_SETUP.md                 # Full ROCm + dual-machine workflow guide
└── CLAUDE.md                      # This file
```

---

## Environment Setup

### Mac (MPS — Apple Silicon)
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ssm3d_env
cd "/Users/benasvaiciulis/Desktop/Projects/Projects/Humanoid Application/ssm-3d-recon"
```

### Linux (ROCm — RX 7700 XT)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # REQUIRED for gfx1101
conda activate ssm3d_env
cd ~/ssm-3d-recon
```

Device is auto-detected everywhere: `cuda` (ROCm shows as cuda) → `mps` → `cpu`.

---

## Current Training State

### Models trained so far

| Model | Sequences | val_AbsRel | Status |
|---|---|---|---|
| v2 (single seq) | freiburg1_desk only | 0.0542 | Superseded |
| v3_single | freiburg1_desk only | 0.0556 | Superseded |
| **v3_multi** | freiburg1 desk+xyz+360 | **0.0541** | ✅ Best so far |
| v4_tum7 | 7 TUM sequences | not started | Next to train |
| v4_diverse | 7 TUM + ARKitScenes | not started | Goal |

Best checkpoint: `outputs/ssm_model_v3_multi/best_model.pt` (epoch 9 of 60)

### Training observations
- Model: 399K params, d_model=128, d_state=64, n_layers=3, seq_len=16
- Overfitting: val loss stopped improving at epoch 9 and bounced 0.054–0.072 for remaining 51 epochs
- Root cause: only 3 freiburg1 sequences (same camera, same building) → not diverse enough
- Fix: train on 7 TUM sequences across 3 different cameras + ARKitScenes (5000+ iPad Pro scenes)

---

## Data Available on Mac

```
data/
├── rgbd_dataset_freiburg1_desk        # 596 frames ✅
├── rgbd_dataset_freiburg1_xyz         # 796 frames ✅
├── rgbd_dataset_freiburg1_360         # 756 frames ✅
├── rgbd_dataset_freiburg1_room        # NEW ✅
├── rgbd_dataset_freiburg2_desk        # NEW ✅ (different camera: fx=520.9)
├── rgbd_dataset_freiburg2_xyz         # NEW ✅
└── rgbd_dataset_freiburg3_long_office_household  # NEW ✅ (different camera: fx=535.4)
```

ARKitScenes: NOT yet downloaded. Requires HuggingFace account + token.
- Accept terms: https://huggingface.co/datasets/apple/ARKitScenes
- Login: `huggingface-cli login`
- Download: `bash scripts/download_datasets.sh arkitscenes`

---

## What Needs to Happen on Linux (Priority Order)

### 1. Set up environment (see LINUX_SETUP.md — full step-by-step)
```bash
# Install ROCm 6.1, then:
conda create -n ssm3d_env python=3.11 -y
conda activate ssm3d_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
pip install -r environment/requirements_linux.txt
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Verify GPU:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Download datasets on Linux (run in parallel, overnight)
```bash
bash scripts/download_datasets.sh tum          # terminal 1
bash scripts/download_datasets.sh arkitscenes  # terminal 2 (need HF token first)
```

### 3. Train v4_diverse (the big one, run overnight with nohup)
```bash
nohup python scripts/03_train_ssm.py \
    --data \
        data/rgbd_dataset_freiburg1_desk \
        data/rgbd_dataset_freiburg1_xyz \
        data/rgbd_dataset_freiburg1_360 \
        data/rgbd_dataset_freiburg1_room \
        data/rgbd_dataset_freiburg2_desk \
        data/rgbd_dataset_freiburg2_xyz \
        data/rgbd_dataset_freiburg3_long_office_household \
        data/arkitscenes/Training/4199 \
        data/arkitscenes/Training/4204 \
        data/arkitscenes/Training/4175 \
        data/arkitscenes/Training/4178 \
        data/arkitscenes/Training/4181 \
    --epochs 60 \
    --d_model 128 \
    --d_state 64 \
    --out outputs/ssm_model_v4_diverse \
    > outputs/train_v4.log 2>&1 &

tail -f outputs/train_v4.log
watch -n 2 rocm-smi   # in another terminal
```

### 4. Evaluate when done
```bash
python scripts/04_evaluate.py \
    --data data/rgbd_dataset_freiburg1_desk \
    --checkpoint outputs/ssm_model_v4_diverse/best_model.pt \
    --out outputs/eval_v4_diverse
```

### 5. Copy best checkpoint back to Mac
```bash
# Replace MAC_IP with your Mac's local IP (System Settings → Wi-Fi → IP)
scp outputs/ssm_model_v4_diverse/best_model.pt \
    benasvaiciulis@MAC_IP:"/Users/benasvaiciulis/Desktop/Projects/Projects/Humanoid Application/ssm-3d-recon/outputs/ssm_model_v4_diverse/best_model.pt"
```

---

## Remaining Submission Tasks

- [ ] Record phone video of room (good lighting, slow movement, textured scene)
- [ ] Run `scripts/00_process_video.py` on phone video → `data/phone_scene`
- [ ] Run TSDF fusion on phone video with best checkpoint → `outputs/mesh_phone`
- [ ] Generate depth comparison video: `scripts/06_render_video.py`
- [ ] Generate mesh screenshots / glb for README example outputs
- [ ] Push to public GitHub repo
- [ ] Fill in application form with repo URL

---

## Key Technical Decisions (don't change these)

**S5 SSM architecture** (`src/models/s5.py`):
- Diagonal state matrix A (not full) for O(N) state ops
- ZOH (Zero-Order Hold) discretisation: exact not approximate
- HiPPO-LegS initialisation: designed for long-range memory
- Parallel scan for training (fast), recurrent for inference (O(1) memory)

**DepthRefinementSSM** (`src/models/temporal_depth.py`):
- Input: DAV2 depth hint [1,H,W] + RGB [3,H,W] per frame → total 4 channels
- CNN encoder: 4ch → 32ch feature map, downsampled 4×
- S5Stack: 3 layers, processes flattened spatial features as sequence
- Output: `depth_out = hint + Tanh(output) * 0.2` — bounded residual, never changes depth by >20%

**Hybrid depth for TSDF** (`scripts/05_tsdf_fusion.py`):
- TUM/ScanNet: where GT depth valid → use GT directly. Where GT=0 (sensor holes ~30% of pixels) → use SSM output scaled per-frame to match GT median. 
- Phone video (no GT): use COLMAP global metric scale × SSM output

**Mesh post-processing** (`scripts/05_tsdf_fusion.py`):
- Largest connected component filter (removes floating shards)
- Statistical outlier removal (nb_neighbors=20, std_ratio=2.0)
- Laplacian smoothing (3 iterations, lambda=0.5)

---

## Known Issues / Gotchas

- `conda: command not found` in new terminal → run `source ~/miniconda3/etc/profile.d/conda.sh` first
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` must be set before every Linux training session
- Open3D TSDF is CPU-only (no GPU) — this is fine, it's fast enough
- Mac M4 offscreen renderer (EGL) fails → `06_render_video.py` falls back to 8 static screenshots instead of turntable video
- ARKitScenes requires HuggingFace login — `huggingface-cli login` then re-run download
- Never run comments as separate zsh commands (`# comment` causes zsh errors) — paste only the actual commands
