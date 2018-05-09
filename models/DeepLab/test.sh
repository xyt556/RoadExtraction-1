#!/bin/bash
#
# This script is used to generate tensorboard graph.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./vis.sh
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


# Visualize the results.
python "${WORK_DIR}"/test.py \
  --logtostderr \
  --vis_split="test" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=1024 \
  --vis_crop_size=1024 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${ROAD_DATASET}" \
  --max_number_of_iterations=100
