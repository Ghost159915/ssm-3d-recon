#!/usr/bin/env bash
# ============================================================
# scripts/01_run_colmap.sh
# ============================================================
# Run COLMAP SfM to extract camera poses from extracted frames.
#
# NOTE: For TUM RGB-D dataset, ground truth poses are already
#   provided in groundtruth.txt — you do NOT need to run COLMAP.
#   This script is here for when you want to use your OWN video
#   (phone footage, etc.) and need to estimate poses.
#
# Usage (TUM — SKIP THIS):
#   The dataset already has poses. Use TUMDataset directly.
#
# Usage (your own video):
#   1. Extract frames first:
#        ffmpeg -i my_video.mp4 -q:v 2 -vf fps=5 frames/%06d.jpg
#
#   2. Run COLMAP:
#        bash scripts/01_run_colmap.sh frames/ colmap_output/
#
#   3. Poses are written to:
#        colmap_output/sparse/0/images.bin  (camera-to-world)
#
#   4. Use pycolmap in Python to read them back.
# ============================================================

set -e

IMAGE_DIR="${1:-frames/}"
OUTPUT_DIR="${2:-colmap_output/}"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Usage: bash 01_run_colmap.sh <image_dir> <output_dir>"
    echo "Example: bash 01_run_colmap.sh frames/ colmap_output/"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/sparse"
mkdir -p "${OUTPUT_DIR}/database"

echo "==> Image dir : ${IMAGE_DIR}"
echo "==> Output dir: ${OUTPUT_DIR}"
echo ""

# Check COLMAP is installed
if ! command -v colmap &>/dev/null; then
    echo "ERROR: COLMAP not found."
    echo "Install:"
    echo "  Mac:   brew install colmap"
    echo "  Linux: sudo apt install colmap   (or build from source)"
    exit 1
fi

DATABASE="${OUTPUT_DIR}/database.db"

# Feature extraction
echo "==> Step 1: Feature extraction..."
colmap feature_extractor \
    --database_path "${DATABASE}" \
    --image_path    "${IMAGE_DIR}" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 0

# Feature matching
echo "==> Step 2: Exhaustive feature matching..."
colmap exhaustive_matcher \
    --database_path "${DATABASE}" \
    --SiftMatching.use_gpu 0

# Sparse reconstruction (SfM)
echo "==> Step 3: Sparse reconstruction..."
colmap mapper \
    --database_path  "${DATABASE}" \
    --image_path     "${IMAGE_DIR}" \
    --output_path    "${OUTPUT_DIR}/sparse"

echo ""
echo "==> COLMAP done."
echo "    Sparse model: ${OUTPUT_DIR}/sparse/0/"
echo ""
echo "==> Read poses in Python with pycolmap:"
echo "    import pycolmap"
echo "    recon = pycolmap.Reconstruction('${OUTPUT_DIR}/sparse/0')"
echo "    for img_id, img in recon.images.items():"
echo "        # img.cam_from_world → camera-to-world pose"
echo "        pose = img.cam_from_world.matrix()"
