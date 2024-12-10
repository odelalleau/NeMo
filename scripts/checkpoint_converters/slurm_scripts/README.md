# `launch_conversion_job` script

This script is designed for converting HuggingFace models into either NeMo or Megatron-LM formats. It supports both homogeneous and heterogeneous model conversion. The script creates an exclusive 1-node SLURM job for the conversion process and can convert 405B models within the job's 4-hour time limit. 

## Requirements

1. **Environment Setup**:
   - The script requires our custom forks of NeMo and Megatron-LM:
     - NeMo fork: [nemo-aim](https://gitlab-master.nvidia.com/deci/research/nemo-aim)
     - Megatron-LM fork: [megatron-lm-aim](https://gitlab-master.nvidia.com/deci/research/megatron-lm-aim)
        - (verified with commit `ec43d612`)
   - Both repositories must be on a branch that supports heterogeneous models.
   - It also requires modelopt with Asha's branch:
     - modelopt repo: [modelopt](https://gitlab-master.nvidia.com/omniml/modelopt). 
       - branch aanoosheh/pytorch-pp-teacher


2. **Directory Structure**:
   - Place modelopt + the forks of NeMo and Megatron-LM in the same parent directory:
     ```
     parent_directory/
     ├── nemo-aim/
     ├── megatron-lm-aim/
     └── modelopt/
     ```

## Usage

### Command-Line Options

- `--hf_checkpoint`: Path to the HuggingFace checkpoint directory.
- `--output_dir`: Directory where the converted model will be saved.
- `--output_format`: Specifies the format in which the converted checkpoint will be saved. Options are nemo or megatron
- `--homogeneous_model`: (Optional) Use the original conversion scripts for homogeneous models. Exclude this flag for heterogeneous models.
- `--megatron_pp_size` pipeline parallel size when saving the checkpoint in megatron format (Default: 1)
- `--megatron_tp_size` tensor parallel size when saving the checkpoint in megatron format (Default: 1)

### Examples

#### Convert a heterogeneous HuggingFace model to Megatron format
```bash
bash scripts/checkpoint_converters/slurm_scripts/launch_conversion_job.sh \
    --hf_checkpoint /lustre/fsw/portfolios/coreai/users/itlevy/hf_repos/RC12_3_new \
    --output_dir /lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/models/megatron_conversion/RC12_3 \
    --output_format megatron \
    --megatron_pp_size 2 \
    --megatron_tp_size 1
```
#### Convert a homogeneous HuggingFace model to a .nemo file
```bash
bash scripts/checkpoint_converters/slurm_scripts/launch_conversion_job.sh \
    --hf_checkpoint /lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/models/meta-llama/Meta-Llama-3.1-70B-Instruct-HF \
    --output_dir /lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/models/megatron_conversion/Llama-3.1-70B-Instruct-NeMo \
    --output_format nemo \
    --homogeneous_model
```
