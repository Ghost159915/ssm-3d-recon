#!/usr/bin/env bash
# scripts/download_datasets.sh — download training datasets
# Usage: bash scripts/download_datasets.sh [tum|arkitscenes|all]
set -e
mkdir -p data

TARGET="${1:-all}"

download_tum() {
    echo ""
    echo "=== TUM RGB-D sequences ==="
    BASE="https://cvg.cit.tum.de/rgbd/dataset"

    download_seq() {
        local name="$1"
        local url="$2"
        local out="data/$name"
        if [ -d "$out" ]; then
            echo "  [skip] $name already exists"
            return
        fi
        echo "  Downloading $name ..."
        curl -L "$url" | tar xz -C data/
        echo "  Done: $out"
    }

    # freiburg1 — office desk, 640x480, Kinect v1
    download_seq "rgbd_dataset_freiburg1_desk"  "$BASE/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
    download_seq "rgbd_dataset_freiburg1_xyz"   "$BASE/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
    download_seq "rgbd_dataset_freiburg1_360"   "$BASE/freiburg1/rgbd_dataset_freiburg1_360.tgz"
    download_seq "rgbd_dataset_freiburg1_room"  "$BASE/freiburg1/rgbd_dataset_freiburg1_room.tgz"

    # freiburg2 — different camera (fx=520.9), longer sequences
    download_seq "rgbd_dataset_freiburg2_desk"  "$BASE/freiburg2/rgbd_dataset_freiburg2_desk.tgz"
    download_seq "rgbd_dataset_freiburg2_xyz"   "$BASE/freiburg2/rgbd_dataset_freiburg2_xyz.tgz"

    # freiburg3 — most diverse (household objects, long trajectory)
    download_seq "rgbd_dataset_freiburg3_long_office_household" \
        "$BASE/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"

    echo ""
    echo "TUM done. Data in data/rgbd_dataset_*"
}

download_arkitscenes() {
    echo ""
    echo "=== ARKitScenes (iPad Pro LiDAR) ==="

    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo "  Installing huggingface_hub..."
        python -m pip install huggingface_hub --quiet
    fi

    mkdir -p data/arkitscenes

    python - <<'PYEOF'
from huggingface_hub import snapshot_download

# 10 diverse scenes: living rooms, bedrooms, kitchens, offices, bathrooms
scenes = ["4199", "4204", "4175", "4178", "4181",
          "4184", "4187", "4190", "4193", "4196"]

patterns = [f"*Training/{sid}/*" for sid in scenes]
# skip heavy high-res and raw sensor files
ignore  = ["*highres*", "*.sens", "*annotations*"]

print(f"Downloading {len(scenes)} ARKitScenes scenes (lowres split)...")
print(f"Scenes: {scenes}")

snapshot_download(
    repo_id="apple/ARKitScenes",
    repo_type="dataset",
    allow_patterns=patterns,
    ignore_patterns=ignore,
    local_dir="data/arkitscenes",
)
print("Done. Scenes in data/arkitscenes/Training/")
PYEOF
}

case "$TARGET" in
    tum)         download_tum ;;
    arkitscenes) download_arkitscenes ;;
    all)
        download_tum
        download_arkitscenes
        ;;
    *)
        echo "Usage: $0 [tum|arkitscenes|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Ready to train with diverse data ==="
echo ""
echo "python scripts/03_train_ssm.py \\"
echo "    --data \\"
echo "        data/rgbd_dataset_freiburg1_desk \\"
echo "        data/rgbd_dataset_freiburg1_xyz \\"
echo "        data/rgbd_dataset_freiburg1_360 \\"
echo "        data/rgbd_dataset_freiburg1_room \\"
echo "        data/rgbd_dataset_freiburg2_desk \\"
echo "        data/rgbd_dataset_freiburg2_xyz \\"
echo "        data/rgbd_dataset_freiburg3_long_office_household \\"
echo "        data/arkitscenes/Training/4199 \\"
echo "        data/arkitscenes/Training/4204 \\"
echo "        data/arkitscenes/Training/4175 \\"
echo "        data/arkitscenes/Training/4178 \\"
echo "        data/arkitscenes/Training/4181 \\"
echo "    --epochs 60 \\"
echo "    --out outputs/ssm_model_v4_diverse"
