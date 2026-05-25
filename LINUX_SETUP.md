# Mac → Linux Setup Guide
### RX 7700 XT + ROCm | Dual-machine workflow for ssm-3d-recon

---

## Overview

The code runs on both machines with zero changes — the device is auto-detected:
- **Mac** → MPS (Apple Silicon GPU) — development, quick tests, inference
- **Linux** → ROCm (RX 7700 XT) — heavy training, large datasets, ~7–8× faster

Code stays in sync via Git. Data and checkpoints stay on each machine (too large to sync).

---

## Part 1 — Push Code to GitHub (Mac, one-time, ~5 min)

### 1.1 Create a GitHub repo

Go to https://github.com/new and create a repo called `ssm-3d-recon` (private is fine).

### 1.2 Push from Mac

Open a terminal on your Mac, `cd` into the project:

```bash
cd "/Users/benasvaiciulis/Desktop/Projects/Projects/Humanoid Application/ssm-3d-recon"
```

Initialise git if not already done:

```bash
git init
git add .
git commit -m "Initial commit — full pipeline"
```

Connect to GitHub and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/ssm-3d-recon.git
git branch -M main
git push -u origin main
```

### 1.3 Add a .gitignore so large files don't get committed

```bash
cat > .gitignore << 'EOF'
# Large data — stays on each machine
data/
outputs/
*.npy
*.pt
*.ply
*.glb
*.mp4

# Python
__pycache__/
*.pyc
.env
ssm3d_env/

# Mac
.DS_Store
EOF

git add .gitignore
git commit -m "Add .gitignore"
git push
```

---

## Part 2 — Set Up the Linux Machine (~30–45 min, one-time)

### 2.1 Update the system

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl python3-pip python3-venv \
                    ffmpeg colmap libgl1-mesa-glx
```

### 2.2 Install ROCm 6.1 (AMD GPU driver + compute stack)

```bash
# Download the ROCm installer
wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
sudo dpkg -i amdgpu-install_6.1.60101-1_all.deb
sudo apt update

# Install ROCm (takes 10–15 min)
sudo amdgpu-install --usecase=rocm --no-dkms -y

# Add yourself to the GPU groups
sudo usermod -a -G render,video $USER

# IMPORTANT: log out and back in (or reboot) for groups to take effect
sudo reboot
```

After reboot, verify ROCm sees the GPU:

```bash
rocm-smi
# Should show: RX 7700 XT, temperature, memory usage
```

### 2.3 Add the RX 7700 XT workaround to your shell

The RX 7700 XT (`gfx1101`) isn't in ROCm's official list yet, but works when told to behave like a 7900 series (`gfx1100`). Add this to `~/.bashrc` permanently:

```bash
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc
```

### 2.4 Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2.5 Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ssm-3d-recon.git
cd ssm-3d-recon
```

### 2.6 Create the Python environment

```bash
conda create -n ssm3d_env python=3.11 -y
conda activate ssm3d_env

# PyTorch with ROCm backend (this is the critical step)
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/rocm6.1

# Everything else
pip install -r environment/requirements_linux.txt

# HuggingFace for dataset downloads
pip install huggingface_hub
```

### 2.7 Verify the GPU works

```bash
conda activate ssm3d_env

python - << 'EOF'
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")   # ROCm shows as CUDA
print(f"Device count    : {torch.cuda.device_count()}")
print(f"Device name     : {torch.cuda.get_device_name(0)}")

# Quick compute test
a = torch.randn(1000, 1000, device="cuda")
b = torch.randn(1000, 1000, device="cuda")
c = a @ b
print(f"Matrix multiply : OK  (result shape {c.shape})")
EOF
```

Expected output:
```
PyTorch version : 2.x.x+rocm6.1
CUDA available  : True
Device name     : AMD Radeon RX 7700 XT
Matrix multiply : OK  (result shape torch.Size([1000, 1000]))
```

If `CUDA available` is False, try:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # make sure this is set
rocm-smi --showproductname               # verify GPU is visible
```

---

## Part 3 — Download Datasets on Linux (~2–8 hrs depending on internet)

Run each in a separate terminal or use `&` to run in background.

### 3.1 TUM RGB-D (no registration, ~3 GB)

```bash
cd ~/ssm-3d-recon
conda activate ssm3d_env
bash scripts/download_datasets.sh tum
```

Downloads 7 sequences: freiburg1 (desk, xyz, 360, room), freiburg2 (desk, xyz), freiburg3 (long office household).

### 3.2 ARKitScenes — iPad Pro LiDAR (no registration, ~5–10 GB)

```bash
bash scripts/download_datasets.sh arkitscenes
```

Downloads 10 diverse indoor scenes (living rooms, bedrooms, kitchens, offices, bathrooms).

### 3.3 NYU Depth V2 (no registration, ~32 GB raw, ~3 GB if you take a subset)

```bash
mkdir -p data/nyu_depth_v2

# Full raw dataset
wget -P data/nyu_depth_v2 \
    http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

# Or just the pre-split subset (much smaller, enough for training):
wget -P data/nyu_depth_v2 \
    http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
```

> NYU dataset loader coming in next update — for now focus on TUM + ARKitScenes.

### 3.4 ScanNet (requires registration — sign up now, takes 1–2 days)

```
1. Go to: https://www.scan-net.org/
2. Fill in the Terms of Use form
3. You'll receive a download-scannet.py script by email
4. Once you have it:

   for id in scene0000_00 scene0001_00 scene0002_00 scene0003_00 \
             scene0004_00 scene0005_00 scene0006_00 scene0007_00 \
             scene0008_00 scene0009_00; do
       python download-scannet.py -o data/scannet --id $id --type .sens
   done

5. Extract each .sens:
   cd ScanNet/SensReader/python
   for sens in ~/ssm-3d-recon/data/scannet/scene*/*.sens; do
       scene=$(dirname $sens)
       python reader.py --filename $sens --output_path $scene \
           --export_color_images --export_depth_images \
           --export_poses --export_intrinsics
   done
```

---

## Part 4 — Training on Linux

### 4.1 Quick sanity check first (~2 min)

Make sure the code runs at all before launching overnight training:

```bash
conda activate ssm3d_env
cd ~/ssm-3d-recon

python scripts/03_train_ssm.py \
    --data data/rgbd_dataset_freiburg1_desk \
    --epochs 2 \
    --max_frames 50 \
    --out outputs/sanity_check
```

You should see DAV2 inference run, then 2 training epochs complete. If it crashes, fix it before starting the big run.

### 4.2 Full diverse training (run overnight)

```bash
conda activate ssm3d_env
cd ~/ssm-3d-recon

# Use nohup so it keeps running if you close the terminal
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
    --seq_len 16 \
    --out outputs/ssm_model_v4_diverse \
    > outputs/train_v4.log 2>&1 &

echo "Training PID: $!"
echo "Watch progress with: tail -f outputs/train_v4.log"
```

Monitor progress:

```bash
tail -f outputs/train_v4.log
```

### 4.3 GPU monitoring while training

Open another terminal:

```bash
watch -n 2 rocm-smi
```

You should see GPU utilisation at 80–100% during training. If it's low (< 50%), the bottleneck is CPU data loading — increase batch size.

---

## Part 5 — Evaluation on Linux

```bash
conda activate ssm3d_env
cd ~/ssm-3d-recon

python scripts/04_evaluate.py \
    --data data/rgbd_dataset_freiburg1_desk \
    --checkpoint outputs/ssm_model_v4_diverse/best_model.pt \
    --out outputs/eval_v4_diverse
```

Expected output: AbsRel, RMSE, δ<1.25 metrics printed and saved to `outputs/eval_v4_diverse/`.

---

## Part 6 — TSDF Fusion & Mesh on Linux

```bash
python scripts/05_tsdf_fusion.py \
    --data data/rgbd_dataset_freiburg1_desk \
    --mode ssm \
    --checkpoint outputs/ssm_model_v4_diverse/best_model.pt \
    --voxel_size 0.02 \
    --sdf_trunc 0.04 \
    --out outputs/mesh_v4
```

Output: `outputs/mesh_v4/scene_ssm.ply` and `.glb` — open `.glb` at https://3dviewer.net

---

## Part 7 — Copy Checkpoint Back to Mac

The checkpoint is the only thing you need to transfer — it's small (~6 MB).

### Option A — SCP over local network (fastest)

Find your Mac's IP:
```
# On Mac: System Settings → Wi-Fi → Details → IP Address
# e.g. 192.168.1.42
```

From Linux:
```bash
scp outputs/ssm_model_v4_diverse/best_model.pt \
    benasvaiciulis@192.168.1.42:"/Users/benasvaiciulis/Desktop/Projects/Projects/Humanoid Application/ssm-3d-recon/outputs/ssm_model_v4_diverse/best_model.pt"
```

### Option B — Push to GitHub as a release asset

```bash
# Install GitHub CLI
sudo apt install gh -y
gh auth login

# Upload checkpoint as a release
gh release create v4-diverse \
    outputs/ssm_model_v4_diverse/best_model.pt \
    --title "v4 diverse model" \
    --notes "Trained on TUM + ARKitScenes, val_AbsRel=X.XXXX"
```

Then on Mac:
```bash
gh release download v4-diverse
```

### Option C — Google Drive / Dropbox

Upload from Linux browser, download on Mac. Slow but no setup.

---

## Part 8 — Ongoing Workflow (keeping both machines in sync)

### When you change code on Mac:

```bash
# Mac
cd "/Users/benasvaiciulis/Desktop/Projects/Projects/Humanoid Application/ssm-3d-recon"
git add -A
git commit -m "describe what changed"
git push
```

```bash
# Linux — pull the changes
cd ~/ssm-3d-recon
git pull
```

### When training finishes on Linux:

```bash
# Linux — copy checkpoint to Mac via scp (see Part 7)
# or push as GitHub release
```

### When you record a phone video on Mac:

```bash
# Mac — process with COLMAP
conda activate ssm3d_env
python scripts/00_process_video.py \
    --video my_room.mp4 \
    --out data/phone_scene

# Copy the processed scene to Linux for high-quality mesh
scp -r data/phone_scene \
    user@LINUX_IP:~/ssm-3d-recon/data/phone_scene

# Linux — fuse with the v4 model
python scripts/05_tsdf_fusion.py \
    --data data/phone_scene \
    --data-type colmap \
    --mode ssm \
    --checkpoint outputs/ssm_model_v4_diverse/best_model.pt \
    --out outputs/mesh_phone

# Copy mesh back to Mac
scp outputs/mesh_phone/scene_ssm.glb \
    benasvaiciulis@MAC_IP:"/Users/benasvaiciulis/Desktop/Projects/Projects/Humanoid Application/ssm-3d-recon/outputs/mesh_phone/"
```

---

## Quick Reference — All Commands

### Mac (development)
```bash
# Activate env
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ssm3d_env

# Train (slow but fine for testing)
python scripts/03_train_ssm.py --data data/rgbd_dataset_freiburg1_desk --epochs 5 --out outputs/test

# Evaluate
python scripts/04_evaluate.py --checkpoint outputs/ssm_model_v3_multi/best_model.pt

# Process phone video
python scripts/00_process_video.py --video my_room.mp4 --out data/phone_scene

# Fuse mesh
python scripts/05_tsdf_fusion.py --data data/rgbd_dataset_freiburg1_desk --mode ssm \
    --checkpoint outputs/ssm_model_v3_multi/best_model.pt
```

### Linux (training powerhouse)
```bash
# Activate env (also sets ROCm workaround)
conda activate ssm3d_env
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Full overnight training
nohup python scripts/03_train_ssm.py \
    --data data/rgbd_dataset_freiburg1_desk data/arkitscenes/Training/4199 [etc] \
    --epochs 60 --out outputs/ssm_model_v4_diverse > outputs/train_v4.log 2>&1 &

# Monitor
tail -f outputs/train_v4.log
watch -n 2 rocm-smi

# Evaluate
python scripts/04_evaluate.py --checkpoint outputs/ssm_model_v4_diverse/best_model.pt

# Copy checkpoint to Mac
scp outputs/ssm_model_v4_diverse/best_model.pt benasvaiciulis@MAC_IP:"~/..."
```

---

## Troubleshooting

**`CUDA available: False` on Linux**
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
rocm-smi   # GPU should be listed
groups     # should include 'render' and 'video'
```

**`conda: command not found` in new terminal**
```bash
source ~/miniconda3/etc/profile.d/conda.sh
# Or add to .bashrc permanently:
echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
```

**`pip: command not found`**
```bash
conda activate ssm3d_env   # always activate first
```

**Out of VRAM during training**
```bash
# Reduce batch size (default is 2, try 1)
python scripts/03_train_ssm.py --batch_size 1 ...
# Or reduce image size
python scripts/03_train_ssm.py --img_h 192 --img_w 256 ...
```

**Open3D TSDF slow on Linux**
```bash
# Open3D uses CPU for TSDF — this is normal, GPU not used for fusion
# It'll still be faster on Linux due to CPU being much stronger
```

**ScanNet `.sens` extraction fails**
```bash
pip install opencv-python-headless   # needed by SensReader
```
