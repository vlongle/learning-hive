#!/bin/bash
#SBATCH --output=slurm_outs/modmod/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=1-8

declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")

# Calculate index for seeds and num_shared_module based on SLURM_ARRAY_TASK_ID
seed_index=$(((SLURM_ARRAY_TASK_ID - 1) % 8)) # Cycle through seeds every 8 iterations

TRANSFER_DECODER="1"
TRANSFER_STRUCTURE="1"
NO_SPARSE_BASIS="1"
SYNC_BASE="1"
DATASET="fashionmnist"

SEED=${seeds[$seed_index]}


srun bash -c "python experiments/modmod_experiments.py --transfer_decoder $TRANSFER_DECODER --transfer_structure $TRANSFER_STRUCTURE --no_sparse_basis $NO_SPARSE_BASIS --sync_base $SYNC_BASE --seed $SEED --dataset $DATASET"
exit 3