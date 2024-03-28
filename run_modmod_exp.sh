#!/bin/bash
#SBATCH --output=slurm_outs/modmod/%A_%a.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=1-7

# declare -a transfer_decoder=("1" "0")
# declare -a transfer_structure=("1" "0")
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")

# Simplified calculation based on SLURM_ARRAY_TASK_ID
# transfer_decoder_index=$((SLURM_ARRAY_TASK_ID / 16)) # Every 16 iterations, switch transfer_decoder
# transfer_structure_index=$(((SLURM_ARRAY_TASK_ID / 8) % 2)) # Switch transfer_structure every 8 iterations, alternating every 16
seed_index=$((SLURM_ARRAY_TASK_ID % 8)) # Cycle through seeds every 8 iterations

# TRANSFER_DECODER=${transfer_decoder[$transfer_decoder_index]}
# TRANSFER_STRUCTURE=${transfer_structure[$transfer_structure_index]}

TRANSFER_DECODER="1"
TRANSFER_STRUCTURE="1"
SEED=${seeds[$seed_index]}

NO_SPARSE_BASIS="1" # Statically set to "1"

srun bash -c "python experiments/modmod_experiments.py --transfer_decoder $TRANSFER_DECODER --transfer_structure $TRANSFER_STRUCTURE --no_sparse_basis $NO_SPARSE_BASIS --sync_base true --seed $SEED"
exit 3

