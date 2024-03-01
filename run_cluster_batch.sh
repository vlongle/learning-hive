#!/bin/bash
#SBATCH --output=slurm_outs/vanilla/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-23 # Adjust this for 24 jobs, ranging from 0 to 23

# Declare the datasets
declare -a datasets=("mnist" "kmnist" "fashionmnist")

# Calculate the dataset index based on SLURM_ARRAY_TASK_ID
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 8))

# Calculate the seed based on SLURM_ARRAY_TASK_ID
SEED=$((SLURM_ARRAY_TASK_ID % 8))

# Assign the dataset based on the calculated index
DATASET=${datasets[$DATASET_INDEX]}

# Run the command with the dataset and seed
srun bash -c "python experiments/experiments.py --seed $SEED --dataset $DATASET"

exit 0
