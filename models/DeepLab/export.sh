#!/bin/bash
#
# This script is used to export the trained checkpoint.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./export.sh
#


# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
DATASET_DIR="datasets"
ROAD_FOLDER="road"
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/export"
ROAD_DATASET="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/tfrecord"

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=24 \
  --output_stride=8 \
  --decoder_output_stride=4 \
  --num_classes=2 \
  --crop_size=256 \
  --crop_size=256 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
