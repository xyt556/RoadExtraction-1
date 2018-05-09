#!/bin/bash
#
# This script is used to run evaluation. This performs eval over
# the full val split (1449 images) andwill take a while.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./valid.sh
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
EXP_FOLDER="exp_xp_step"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/export"
ROAD_DATASET="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/tfrecord"

mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Run
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=512 \
  --eval_crop_size=512 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${ROAD_DATASET}" \
  --max_number_of_evaluations=100
