#!/bin/bash
set -euo pipefail
set -x  # Enable debugging

# make sure conda is not used
unwanted_prefix="/lustre/fsw/portfolios/coreai/users"
export PATH=$(echo "$PATH" | sed -e "s;:$unwanted_prefix[^:]*;;g" -e "s;$unwanted_prefix[^:]*:;;g" -e "s;$unwanted_prefix[^:]*;;g")

pip uninstall pytorch-lightning -y
pip install lightning

cp -R /opt/modelopt /workspace/modelopt
pip uninstall nvidia-modelopt -y
pip install /workspace/modelopt --extra-index-url https://pypi.nvidia.com --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

cd /opt/NeMo

required_env_vars=(
  "HF_HOME" "HF_CHECKPOINT" "OUTPUT_DIR" "OUTPUT_FORMAT" "MEGATRON_PP_SIZE" "MEGATRON_TP_SIZE"
)

# Iterate over the list and check each one
for var_name in "${required_env_vars[@]}"; do
    if [ -z "${!var_name}" ]; then  # The exclamation mark is used for indirect reference
        echo "Error: $var_name is not set. Set all of these vars: ${required_env_vars[@]}"
        exit 1
    fi
done

if [[ "${OUTPUT_FORMAT}" == "nemo" ]]; then
    OUTPUT_FORMAT_NAME="NeMo"
    if [[ $MEGATRON_TP_SIZE -ne 1 || $MEGATRON_PP_SIZE -ne 1 ]]; then
        echo "Assertion failed: MEGATRON_TP_SIZE or MEGATRON_PP_SIZE variables are not equal to 1, which is a must for nemo conversion"
        exit 1
    fi
else
    OUTPUT_FORMAT_NAME="Megatron"
fi

FINAL_OUTPUT_DIR="${OUTPUT_DIR}/${OUTPUT_FORMAT_NAME}"
mkdir -p ${FINAL_OUTPUT_DIR}
# OUTPUT_FILE="${FINAL_OUTPUT_DIR}/model.nemo"
INTERMEDIATE_OUTPUT_DIR="${OUTPUT_DIR}/files_per_weight"

if [[ ${HOMOGENEOUS_MODEL:-} ]]; then
    echo "Converting homogeneous model"
    SAVE_DICT_SCRIPT="scripts/checkpoint_converters/convert_llama_hf_to_nemo_save_dict.py"
    SAVE_DICT_FINAL_DIR_STR=""
    NEMO_LOAD_SCRIPT="scripts/checkpoint_converters/convert_llama_hf_to_nemo_load.py"
else
    echo "Converting heterogeneous model"
    SAVE_DICT_SCRIPT="scripts/checkpoint_converters/convert_heterogeneous_llama_hf_to_nemo_save_dict.py"
    SAVE_DICT_FINAL_DIR_STR="--final_nemo_path ${FINAL_OUTPUT_DIR}"
    NEMO_LOAD_SCRIPT="scripts/checkpoint_converters/convert_heterogeneous_llama_hf_to_nemo_load.py"
fi

if [ -d "$INTERMEDIATE_OUTPUT_DIR" ] && [ "$(ls -A "$INTERMEDIATE_OUTPUT_DIR")" ]; then
    echo "$INTERMEDIATE_OUTPUT_DIR is not empty, skipping saving individual weight files"
    cp ${HF_CHECKPOINT}/config.json ${FINAL_OUTPUT_DIR}/ 
else
    mkdir -p ${INTERMEDIATE_OUTPUT_DIR}

    python $SAVE_DICT_SCRIPT \
        --input_name_or_path ${HF_CHECKPOINT} \
        --output_path ${INTERMEDIATE_OUTPUT_DIR} \
        ${SAVE_DICT_FINAL_DIR_STR} \
        --precision bf16 \
        --apply_rope_scaling True

    echo "Intermediate files saved to ${INTERMEDIATE_OUTPUT_DIR}"
fi


if [[ "$OUTPUT_FORMAT" == "nemo" ]]; then
    NEMO_OUTPUT_FILE="${FINAL_OUTPUT_DIR}/model.nemo"
    python $NEMO_LOAD_SCRIPT \
        --input_name_or_path ${HF_CHECKPOINT} \
        --input_state_dict ${INTERMEDIATE_OUTPUT_DIR} \
        --output_path ${NEMO_OUTPUT_FILE} \
        --precision bf16 \
        --llama31 True \
        --pack_nemo_file
else
    cd /opt/megatron-lm
    python tools/checkpoint/convert.py \
        --bf16 \
        --model-type GPT \
        --loader llama_from_individual_files \
        --saver mcore_heterogeneous \
        --target-tensor-parallel-size ${MEGATRON_TP_SIZE} \
        --target-pipeline-parallel-size ${MEGATRON_PP_SIZE} \
        --load-dir ${HF_CHECKPOINT} \
        --save-dir ${FINAL_OUTPUT_DIR} \
        --tokenizer-model ${HF_CHECKPOINT} \
        --max-queue-size 10 \
        --individual-files-per-weight-dir ${INTERMEDIATE_OUTPUT_DIR}
fi

echo "Model converted to ${OUTPUT_FORMAT} format and saved to ${FINAL_OUTPUT_DIR}"