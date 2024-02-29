#!/bin/bash
#SBATCH --output=slurm_outs/modmod/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-23 # Now 24 jobs in total for each combination of settings

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a transfer_decoder=("0" "1")
declare -a transfer_structure=("0" "1")
declare -a no_sparse_basis=("0" "1")

# Calculate dataset index
dataset_index=$(($SLURM_ARRAY_TASK_ID / 8)) # 8 combinations of transfer and no_sparse_basis settings per dataset
# Calculate transfer_decoder index
transfer_decoder_index=$((($SLURM_ARRAY_TASK_ID / 4) % 2)) # Alternates every 4 jobs
# Calculate transfer_structure index
transfer_structure_index=$((($SLURM_ARRAY_TASK_ID / 2) % 2)) # Alternates every 2 jobs
# Calculate no_sparse_basis index
no_sparse_basis_index=$(($SLURM_ARRAY_TASK_ID % 2)) # Alternates every job

DATASET=${datasets[$dataset_index]}
TRANSFER_DECODER=${transfer_decoder[$transfer_decoder_index]}
TRANSFER_STRUCTURE=${transfer_structure[$transfer_structure_index]}
NO_SPARSE_BASIS=${no_sparse_basis[$no_sparse_basis_index]}

srun bash -c "python experiments/modmod_experiments.py --dataset $DATASET --transfer_decoder $TRANSFER_DECODER --transfer_structure $TRANSFER_STRUCTURE --no_sparse_basis $NO_SPARSE_BASIS"
exit 0
