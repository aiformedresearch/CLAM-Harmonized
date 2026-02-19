#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_clam_single_input.sh <input_dir> <meta_out_dir> <features_out_dir> <slide.(svs|btf|tif|tiff)> <batch_size> [encoder] [target_patch_size] [target_mpp] [step_size] [ref_patch]
#
# Notes:
# - target_patch_size = CNN input tile size (e.g., 224)
# - ref_patch = patch size used for patch extraction under harmonization (e.g., 448)

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <input_dir> <meta_out_dir> <features_out_dir> <slide.(svs|btf|tif|tiff)> <batch_size> [encoder] [target_patch_size] [target_mpp] [step_size] [ref_patch]"
  exit 1
fi

INPUT_DIR="$1"
META_OUT="$2"
FEATURES_OUT="$3"
SLIDE_ARG="$4"
BATCH_SIZE="${5:-800}"
ENCODER="${6:-resnet50_trunc}"
TARGET_PATCH_SIZE="${7:-224}"

# Harmonization params (new, optional)
TARGET_MPP="${8:-0.25}"
STEP_SIZE="${9:-256}"
REF_PATCH="${10:-448}"

# Resolve slide path (allow either absolute/relative path, or just filename under INPUT_DIR)
if [[ -f "$SLIDE_ARG" ]]; then
  SLIDE_PATH="$SLIDE_ARG"
else
  SLIDE_PATH="$INPUT_DIR/$SLIDE_ARG"
fi

if [[ ! -f "$SLIDE_PATH" ]]; then
  echo "ERROR: slide not found: $SLIDE_PATH"
  exit 1
fi

SLIDE_NAME="$(basename "$SLIDE_PATH")"
SLIDE_STEM="${SLIDE_NAME%.*}"
SLIDE_EXT=".${SLIDE_NAME##*.}"
SLIDE_EXT_LOWER="$(echo "$SLIDE_EXT" | tr '[:upper:]' '[:lower:]')"

case "$SLIDE_EXT_LOWER" in
  .svs|.btf|.tif|.tiff) ;;
  *) echo "ERROR: Unsupported slide extension: $SLIDE_EXT_LOWER" ; exit 1 ;;
esac

# Ensure INPUT_DIR matches slide location (needed because slide_id in process list is relative to --source)
SLIDE_DIR="$(dirname "$SLIDE_PATH")"
if [[ "$(realpath "$SLIDE_DIR")" != "$(realpath "$INPUT_DIR")" ]]; then
  INPUT_DIR="$SLIDE_DIR"
fi

mkdir -p \
  "$META_OUT/masks" \
  "$META_OUT/patches" \
  "$META_OUT/stitches" \
  "$META_OUT/csv_files" \
  "$META_OUT/logs"

mkdir -p "$FEATURES_OUT"

PROC_CSV="$META_OUT/single_process_list.csv"
printf "slide_id,process\n%s,1\n" "$SLIDE_NAME" > "$PROC_CSV"

echo "[info] slide=$SLIDE_NAME ext=$SLIDE_EXT_LOWER"
echo "[info] harmonize target_mpp=$TARGET_MPP ref_patch=$REF_PATCH step_size=$STEP_SIZE"
echo "[info] feature extraction encoder=$ENCODER cnn_tile=$TARGET_PATCH_SIZE batch_size=$BATCH_SIZE"

# Patch creation (harmonized)
echo "Starting patch creation (harmonized)"
python3 create_patches_fp.py \
  --source "$INPUT_DIR" \
  --save_dir "$META_OUT" \
  --preset tcga.csv \
  --harmonize \
  --target_mpp "$TARGET_MPP" \
  --patch_size "$REF_PATCH" \
  --step_size "$STEP_SIZE" \
  --seg --patch --stitch \
  --process_list "$(basename "$PROC_CSV")" \
  > "$META_OUT/logs/${SLIDE_STEM}_patch_creation.log" 2>&1

cp -f "$PROC_CSV" \
  "$META_OUT/csv_files/${SLIDE_STEM}_process_list_autogen.csv"

# Feature extraction
echo "Starting feature extraction"
python3 extract_features_fp.py \
  --data_h5_dir "$META_OUT" \
  --data_slide_dir "$INPUT_DIR" \
  --csv_path "$META_OUT/csv_files/${SLIDE_STEM}_process_list_autogen.csv" \
  --feat_dir "$FEATURES_OUT" \
  --slide_ext "$SLIDE_EXT_LOWER" \
  --batch_size "$BATCH_SIZE" \
  --target_patch_size "$TARGET_PATCH_SIZE" \
  --model_name "$ENCODER" \
  > "$META_OUT/logs/${SLIDE_STEM}_feature_extraction.log" 2>&1

echo "Feature extraction completed"
