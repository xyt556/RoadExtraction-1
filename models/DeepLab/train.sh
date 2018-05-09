#!/bin/bash
#
# This script is used: to train road dataset.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./train.sh
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/model_test.py -v

# Go to datasets folder, preprocessed road dataset and convert data to TFrecord.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
#sh convert_road.sh
# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
ROAD_FOLDER="road"
EXP_FOLDER="exp_xp_step"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/${EXP_FOLDER}/export"

mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
# TF_INIT_ROOT="http://download.tensorflow.org/models"
# TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
# cd "${INIT_FOLDER}"
# wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
# tar -xf "${TF_INIT_CKPT}"
# cd "${CURRENT_DIR}"


# Train 50000 iterations.
# model_variant: 'xception_65' (better) or 
# resize_value: for data augmentation
# train_crop_size: could be a little smaller than 512
# learning_policy: 'step' 'poly' 'adam'
# base_learning_rate: 0.007 around for training and 0.0001 around for finetune
# learning_rate_decay_factor: rate to decay the base learning rate, when use policy 'poly'
# learning_power: when use policy 'step'
# weight_decay: usually <= 0.00004, weight for l2 regularization loss
# tf_initial_checkpoint: 'deeplabv3_cityscapes_train'  cityscapes pretrained deeplab
#                        'xception_imagenet'  imagenet pretrained xception 
#                        'deeplabv3_pascal_train_aug'  VOC pretrained deeplab 
# initialize_last_layer: False if use new checkpoint
# last_layers_contain_logits_only: False if include ASPP and decoder as last layer 
# last_layer_gradient_multiplier: boost the gradient of last layers
# train_batch_size: may be out of memory if > 7
# fine_tune_batch_norm: True if batch size > 8, False otherwise
# log_steps: show log, in seconds 
# save_interval_secs: save the model to disk, in seconds
# save_summaries_secs: show in tensorboard, in seconds 
# save_summaries_images: True if save inputs, labels, predictions

ROAD_DATASET="${WORK_DIR}/${DATASET_DIR}/${ROAD_FOLDER}/tfrecord"

# Train 3000 iterations.

python deeplab/train3.py \
    --logtostderr \
    --training_number_of_steps=60000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --learning_policy='step' \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --learning_power=0.9 \
    --learning_rate_decay_factor=0.1 \
    --train_crop_size=512 \
    --train_crop_size=512 \
    --weight_decay=0.00004 \
    --learning_rate_decay_step=12000 \
    --base_learning_rate=0.0002 \
    --fine_tune_batch_norm=False \
    --train_batch_size=4 \
    --dataset="road" \
    --initialize_last_layer=False \
    --last_layer_gradient_multiplier=40 \
    --last_layers_contain_logits_only=False \
    --tf_initial_checkpoint="${INIT_FOLDER}/xception_imagenet/model.ckpt" \
    --train_logdir=${TRAIN_LOGDIR}\
    --dataset_dir=${ROAD_DATASET}
