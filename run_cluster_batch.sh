#!/bin/bash
#SBATCH --output=slurm_outs/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-5 # Adjust this for 6 jobs, ranging from 0 to 5

# Declare the datasets
declare -a datasets=("mnist" "kmnist" "fashionmnist")

# Options for no_sparse_basis
declare -a no_sparse_basis=("0" "1")

# Calculate the dataset index and no_sparse_basis option based on SLURM_ARRAY_TASK_ID
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
NO_SPARSE_BASIS_INDEX=$((SLURM_ARRAY_TASK_ID % 2))

# Assign the dataset and no_sparse_basis option based on the calculated indices
DATASET=${datasets[$DATASET_INDEX]}
NO_SPARSE_BASIS=${no_sparse_basis[$NO_SPARSE_BASIS_INDEX]}

# Run the command with the dataset and no_sparse_basis option
srun bash -c "python experiments/experiments.py --seed $SLURM_ARRAY_TASK_ID --dataset $DATASET --no_sparse_basis $NO_SPARSE_BASIS"

exit 0