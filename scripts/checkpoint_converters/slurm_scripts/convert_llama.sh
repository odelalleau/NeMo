#!/bin/bash
set -euo pipefail
set -x  # Enable debugging

# make sure conda is not used
unwanted_prefix="/lustre/fsw/portfolios/coreai/users"
export PATH=$(echo "$PATH" | sed -e "s;:$unwanted_prefix[^:]*;;g" -e "s;$unwanted_prefix[^:]*:;;g" -e "s;$unwanted_prefix[^:]*;;g")

pip install lightning
cd /opt/NeMo


required_env_vars=(
  "HF_HOME" "HF_CHECKPOINT" "OUTPUT_DIR"
)

# Iterate over the list and check each one
for var_name in "${required_env_vars[@]}"; do

    if [ -z "${!var_name}" ]; then  # The exclamation mark is used for indirect reference
        echo "Error: $var_name is not set. Set all of these vars: ${required_env_vars[@]}"
        exit 1
    fi
done

if [[ ${PACK_NEMO_FILE:-} ]]; then
    OUTPUT_FORMAT="NeMo"
    PACK_FLAG="--pack_nemo_file"
else
    OUTPUT_FORMAT="Megatron"
    PACK_FLAG=""
fi

NEMO_OUTPUT_DIR="${OUTPUT_DIR}/${OUTPUT_FORMAT}"
mkdir -p ${NEMO_OUTPUT_DIR}
NEMO_OUTPUT_FILE="${NEMO_OUTPUT_DIR}/model.nemo"
NEMO_INTERMEDIATE_OUTPUT="${OUTPUT_DIR}/files_per_weight"
mkdir -p ${NEMO_INTERMEDIATE_OUTPUT}

if [[ ${HOMOGENEOUS_MODEL:-} ]]; then
    echo "Converting homogeneous model"
    SAVE_DICT_SCRIPT="scripts/checkpoint_converters/convert_llama_hf_to_nemo_save_dict.py"
    SAVE_DICT_FINAL_DIR_STR=""
    LOAD_SCRIPT="scripts/checkpoint_converters/convert_llama_hf_to_nemo_load.py"
else
    echo "Converting heterogeneous model"
    SAVE_DICT_SCRIPT="scripts/checkpoint_converters/convert_heterogeneous_llama_hf_to_nemo_save_dict.py"
    SAVE_DICT_FINAL_DIR_STR="--final_nemo_path ${NEMO_OUTPUT_FILE}"
    LOAD_SCRIPT="scripts/checkpoint_converters/convert_heterogeneous_llama_hf_to_nemo_load.py"
fi

python $SAVE_DICT_SCRIPT \
    --input_name_or_path $HF_CHECKPOINT \
    --output_path $NEMO_INTERMEDIATE_OUTPUT \
    ${SAVE_DICT_FINAL_DIR_STR} \
    --precision bf16 \
    --apply_rope_scaling True

echo "Intermediate files saved to ${NEMO_INTERMEDIATE_OUTPUT}"

python $LOAD_SCRIPT \
     --input_name_or_path $HF_CHECKPOINT \
     --input_state_dict $NEMO_INTERMEDIATE_OUTPUT \
     --output_path $NEMO_OUTPUT_FILE \
     --precision bf16 \
     --llama31 True \
    ${PACK_FLAG}

echo "Model converted to ${OUTPUT_FORMAT} format and saved to ${NEMO_OUTPUT_DIR}"