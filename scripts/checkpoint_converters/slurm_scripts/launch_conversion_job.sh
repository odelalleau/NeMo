#!/bin/bash

source $(dirname $0)/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('--hf_checkpoint', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--pack_nemo_file', action='store_true')
parser.add_argument('--homogeneous_model', action='store_true')
EOF


MY_DIR=/lustre/fsw/portfolios/coreai/users/${USER}
NEMO_DIR=${MY_DIR}/repos/megatron/nemo-aim
MLM_DIR=${MY_DIR}/repos/megatron/megatron-lm-aim

CONTAINER_REPOS_DIR=/opt
LOGS_DIR=$MY_DIR/slurm_logs
COMMAND_FILE="scripts/checkpoint_converters/slurm_scripts/cd_and_execute_recipe.sh"
IMAGE_PATH=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/docker_images/nemo/24_09.sqsh
GPUS_PER_NODE=1
DURATION=4

JOB_TYPE="convert_hf_checkpoint"
MODEL_NAME=$(basename "$OUTPUT_DIR")
JOB_NAME="convert_${MODEL_NAME}"

RECIPE_FILE="scripts/checkpoint_converters/slurm_scripts/convert_llama.sh"
ENV_VARS_ARRAY=(
  "RECIPE_FILE=$RECIPE_FILE"
  "HF_HOME=$MY_DIR/hf_cache"
  "HF_CHECKPOINT=$HF_CHECKPOINT"
  "OUTPUT_DIR=$OUTPUT_DIR"
  "PACK_NEMO_FILE=$PACK_NEMO_FILE"
  "HOMOGENEOUS_MODEL=$HOMOGENEOUS_MODEL"
)
# Join the array into a single string with commas
ENV_VARS=$(IFS=,; echo "${ENV_VARS_ARRAY[*]}")

submit_job \
    --duration=${DURATION} \
    --gpu=${GPUS_PER_NODE} \
    --nodes=1 \
    --exclusive \
    --skip_image_check \
    --image=${IMAGE_PATH} \
    --name=${JOB_NAME} \
    --logroot=${LOGS_DIR}/${JOB_TYPE} \
    --mounts ${NEMO_DIR}:${CONTAINER_REPOS_DIR}/NeMo,${MLM_DIR}:${CONTAINER_REPOS_DIR}/megatron-lm,/lustre:/lustre \
    --email_mode='always' \
    --notify_on_start  \
    --notification_method='slack' \
    --setenv=${ENV_VARS} \
    --partition batch \
    --command ${COMMAND_FILE}
