#!/usr/bin/env bash
set -euo pipefail

# Container runtime portability:
# - Some systems use 'apptainer' instead of 'singularity'.
# - If $RUNTIME is not set, auto-detect apptainer first, then singularity.
RUNTIME="${RUNTIME:-}"
if [[ -z "$RUNTIME" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    RUNTIME="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    RUNTIME="singularity"
  else
    echo "ERROR: neither 'apptainer' nor 'singularity' found in PATH."
    echo "Install one of them, or set RUNTIME=/full/path/to/apptainer (or singularity)."
    exit 1
  fi
fi

label_frac_list=(1.0) # use all available data. change to e.g. (0.01 0.1 0.5 1.0) to run multiple fractions sequentially in one go.
GPU=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BASE_DIR="$SCRIPT_DIR"

LOCAL_RESULTS_ROOT="$LOCAL_BASE_DIR/train_eval_output"
LOCAL_TOAD_CODE_DIR="$LOCAL_BASE_DIR/src_train_eval"

LOCAL_SINGULARITY_IMAGE="$LOCAL_BASE_DIR/assets/containers/singularity_train_eval.simg"

# Path to preprocessing_output/FEATURES (contains pt_files/, h5_files/)
# Provide explicitly via --features when running.
LOCAL_FEATURES_DIR="$LOCAL_BASE_DIR/preprocessing_output/FEATURES"

usage() {
  cat <<EOF
Usage:
  bash $(basename "$0") --features /path/to/preprocessing_output/FEATURES [--gpu 0] [--image /path/to/container.simg]

Required:
  --features   Path to the FEATURES folder produced by preprocessing (contains pt_files/, h5_files/)

Optional:
  --gpu        GPU id to expose inside container (default: 0)
  --image      Container image path

Runtime:
  RUNTIME=apptainer   (or RUNTIME=singularity) to force a specific runtime.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --features) LOCAL_FEATURES_DIR="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --image) LOCAL_SINGULARITY_IMAGE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -n "$LOCAL_FEATURES_DIR" ]] || { echo "ERROR: --features is required"; usage; exit 1; }
[[ -d "$LOCAL_FEATURES_DIR" ]] || { echo "ERROR: features dir not found: $LOCAL_FEATURES_DIR"; exit 1; }

[[ -d "$LOCAL_TOAD_CODE_DIR" ]] || { echo "ERROR: missing code dir: $LOCAL_TOAD_CODE_DIR"; exit 1; }
[[ -f "$LOCAL_TOAD_CODE_DIR/run_python_scripts.sh" ]] || { echo "ERROR: missing script: $LOCAL_TOAD_CODE_DIR/run_python_scripts.sh"; exit 1; }

[[ -f "$LOCAL_TOAD_CODE_DIR/create_splits.py" ]] || { echo "ERROR: missing $LOCAL_TOAD_CODE_DIR/create_splits.py"; exit 1; }
[[ -f "$LOCAL_TOAD_CODE_DIR/main_mtl_concat.py" ]] || { echo "ERROR: missing $LOCAL_TOAD_CODE_DIR/main_mtl_concat.py"; exit 1; }
[[ -f "$LOCAL_TOAD_CODE_DIR/eval_mtl_concat.py" ]] || { echo "ERROR: missing $LOCAL_TOAD_CODE_DIR/eval_mtl_concat.py"; exit 1; }

[[ -f "$LOCAL_SINGULARITY_IMAGE" ]] || { echo "ERROR: image not found: $LOCAL_SINGULARITY_IMAGE"; exit 1; }

mkdir -p "$LOCAL_RESULTS_ROOT"

FEATURES_REALPATH="$(readlink -f "$LOCAL_FEATURES_DIR")"
TOAD_CODE_REALPATH="$(readlink -f "$LOCAL_TOAD_CODE_DIR")"
RESULTS_REALPATH="$(readlink -f "$LOCAL_RESULTS_ROOT")"
IMAGE_REALPATH="$(readlink -f "$LOCAL_SINGULARITY_IMAGE")"

echo "=== Train/Eval runs (SEQUENTIAL) ==="
echo "Host code dir:   $TOAD_CODE_REALPATH   (bind-mounted as /app/TOAD_repo)"
echo "Container image: $IMAGE_REALPATH"
echo "Features dir:    $FEATURES_REALPATH"
echo "Results root:    $RESULTS_REALPATH"
echo "GPU:             $GPU"
echo "Runtime:         $RUNTIME"
echo

for LABEL_FRAC in "${label_frac_list[@]}"; do
  LABEL_PCT="$(python3 - <<'PY' "$LABEL_FRAC"
import sys
val = float(sys.argv[1])
lf = val/100.0 if val > 1.0 else val
print(int(round(lf * 100)))
PY
)"
  echo "label frac (input): ${LABEL_FRAC}  | label pct (int): ${LABEL_PCT}"

  LOCAL_RESULTS_folder_name="RESULTS_EXP_${LABEL_PCT}"
  LOCAL_RESULTS_DIR="$LOCAL_RESULTS_ROOT/$LOCAL_RESULTS_folder_name"
  mkdir -p "$LOCAL_RESULTS_DIR"

  LOCAL_SPLITS_DIR="$LOCAL_RESULTS_DIR/splits"
  mkdir -p "$LOCAL_SPLITS_DIR"

  LOCAL_EVAL_RESULTS_DIR="$LOCAL_RESULTS_DIR/eval_results"
  mkdir -p "$LOCAL_EVAL_RESULTS_DIR"

  log_file_name="$LOCAL_RESULTS_DIR/${LOCAL_RESULTS_folder_name}_split_train_eval.log"
  echo "Starting container. Log: ${log_file_name}"

  {
    echo "timestamp: $(date --iso-8601=seconds 2>/dev/null || date)"
    echo "label_frac_input: ${LABEL_FRAC}"
    echo "label_pct_int: ${LABEL_PCT}"
    echo "gpu: ${GPU}"
    echo "runtime: ${RUNTIME}"
    echo "container_image: ${IMAGE_REALPATH}"
    echo "features_dir: ${FEATURES_REALPATH}"
    echo "host_code_dir: ${TOAD_CODE_REALPATH}"
    echo "container_code_dir: /app/TOAD_repo (bind-mounted from host)"
    echo "splits_dir_host: ${LOCAL_SPLITS_DIR}"
    echo "splits_dir_container: /app/TOAD_repo/splits"
    echo "features_container_mount: /app/data/FEATURES"
  } > "${LOCAL_RESULTS_DIR}/run_meta.txt"

  export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU"
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU"

  "$RUNTIME" exec --nv     --bind "$TOAD_CODE_REALPATH:/app/TOAD_repo:ro"     --bind "$LOCAL_EVAL_RESULTS_DIR:/app/TOAD_repo/eval_results"     --bind "$FEATURES_REALPATH:/app/data/FEATURES:ro"     --bind "$LOCAL_RESULTS_DIR:/app/results"     --bind "$LOCAL_SPLITS_DIR:/app/TOAD_repo/splits"     --containall     "$IMAGE_REALPATH"     /bin/bash -lc 'cd /app/TOAD_repo && bash run_python_scripts.sh "$1" "$2"' -- "$LABEL_FRAC" "$GPU"     > "${log_file_name}" 2>&1

  echo "Run completed: ${LOCAL_RESULTS_folder_name}"
  echo
done

echo "All runs completed successfully."
