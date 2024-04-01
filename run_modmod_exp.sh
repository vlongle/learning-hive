#!/bin/bash
#SBATCH --output=slurm_outs/modmod/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=1-24

declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")
declare -a num_shared_module=("2" "3" "4")

# Calculate index for seeds and num_shared_module based on SLURM_ARRAY_TASK_ID
seed_index=$(((SLURM_ARRAY_TASK_ID - 1) % 8)) # Cycle through seeds every 8 iterations
num_shared_module_index=$(((SLURM_ARRAY_TASK_ID - 1) / 8)) # Cycle through num_shared_module every 8 seeds

TRANSFER_DECODER="1"
TRANSFER_STRUCTURE="1"
SEED=${seeds[$seed_index]}
NUM_SHARED_MODULE=${num_shared_module[$num_shared_module_index]}

NO_SPARSE_BASIS="1" # Statically set to "1"

srun bash -c "python experiments/modmod_experiments.py --transfer_decoder $TRANSFER_DECODER --transfer_structure $TRANSFER_STRUCTURE --num_shared_module $NUM_SHARED_MODULE --no_sparse_basis $NO_SPARSE_BASIS --sync_base true --seed $SEED"
exit 3