# !/bin/bash
# Script to preprocess and convert road dataset.
#
# Usage:
#   sh ./convert_road.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_road_data.py
#     - build_road_test_data.py
#     - convert_road.sh
#     - split_road.py
#     + road
#       + raw_data
#         + train`-val
#         + test
#       + data
#         + train
#         + valid
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="."

# Root path for ROAD dataset.
ROAD_ROOT="${WORK_DIR}/road"

# Preprocess data.
INPUT_DIR="${ROAD_ROOT}/raw_data"
OUTPUT_DIR="${ROAD_ROOT}/data"


python "${WORK_DIR}"/split_road.py \
  --root="${INPUT_DIR}" \
  --target_dir="${OUTPUT_DIR}"


# Build TFRecords of trainval dataset.
INPUT_DIR="${OUTPUT_DIR}"
OUTPUT_DIR="${ROAD_ROOT}/tfrecord"

echo "Converting road train_val dataset..."
python "${CURRENT_DIR}"/build_road_data.py \
  --data_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}"


# Build TFRecords of test dataset
INPUT_DIR="${ROAD_ROOT}/raw_data/test"
OUTPUT_DIR="${ROAD_ROOT}/tfrecord"

echo "Converting road test dataset..."
python "${CURRENT_DIR}"/build_road_test_data.py \
  --data_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}"



