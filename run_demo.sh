#!/usr/bin/env bash
# run_demo.sh — convenience wrapper to run the Tiny-ImageNet smoke demo end-to-end
# Usage: ./run_demo.sh [--download-tiny] [--max_images N] [--num_example N] [--create-small-train]

set -euo pipefail
ROOT_DIR="$(pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
DATA_ROOT="${ROOT_DIR}/datasets/tiny-imagenet-200/train"
SMALL_TRAIN="${ROOT_DIR}/datasets/tiny-imagenet-200/small_train"
MAX_IMAGES=${MAX_IMAGES:-200}
NUM_EXAMPLE=${NUM_EXAMPLE:-8}
CREATE_SMALL=${CREATE_SMALL:-0}
DOWNLOAD_TINY=${DOWNLOAD_TINY:-0}

function info(){ echo "[INFO] $*"; }

info "Running demo from ${ROOT_DIR}"

# Activate venv if present
if [ -d "${VENV_DIR}" ]; then
  info "Activating virtualenv at ${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
else
  info "No .venv found. The script will try to install packages into the current Python environment."
fi

info "Installing remaining Python requirements (requirements.txt)"
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  info "No requirements.txt found — please create one or install dependencies manually."
fi

if [ "${DOWNLOAD_TINY}" -ne 0 ]; then
  info "Downloading Tiny-ImageNet (this will download ~260MB)"
  mkdir -p datasets
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O datasets/tiny-imagenet-200.zip
  unzip -q datasets/tiny-imagenet-200.zip -d datasets/
fi

if [ "${CREATE_SMALL}" -ne 0 ]; then
  info "Creating small_train subset at ${SMALL_TRAIN} — copying the first 10 classes"
  mkdir -p "${SMALL_TRAIN}"
  # Copy first 10 class folders
  COUNT=0
  for d in $(ls -1 "${DATA_ROOT}" | head -n 10); do
    if [ -d "${DATA_ROOT}/${d}" ]; then
      cp -r "${DATA_ROOT}/${d}" "${SMALL_TRAIN}/${d}"
      COUNT=$((COUNT+1))
    fi
  done
  info "Copied ${COUNT} classes to small_train"
fi

info "Step 1: compute class-level shap scores (limited to ${MAX_IMAGES})"
python extract_shap.py --data_root "${DATA_ROOT}" --shap_save_root ./utils/heat/class_shap.pkl --max_images ${MAX_IMAGES}

info "Step 2: select example images (num_example=${NUM_EXAMPLE})"
# If small_train exists, prefer that
EXAMPLE_DATA_ROOT="${SMALL_TRAIN}"
if [ ! -d "${EXAMPLE_DATA_ROOT}" ]; then
  EXAMPLE_DATA_ROOT="${DATA_ROOT}"
fi
python example_selection.py --data_root "${EXAMPLE_DATA_ROOT}" --num_example ${NUM_EXAMPLE} --num_act 1 --save_root ./utils --img_save_root ./images/example_val_fc

info "Step 3: concept matching (CLIP)"
python concept_matching.py --img_save_root ./images/example_val_fc --num_example ${NUM_EXAMPLE} --data 1

info "Step 4: generate heatmaps"
python image_heatmap.py --example_root ./images/example_val_fc --num_example ${NUM_EXAMPLE} --heatmap_save_root ./heatmap --util_root ./utils --map_root ./heatmap_info

info "Demo finished. Outputs are under ./utils, ./images/example_val_fc, ./heatmap, and ./heatmap_info"

