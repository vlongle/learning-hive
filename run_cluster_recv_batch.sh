#!/bin/bash
#SBATCH --output=slurm_outs/fl/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-11 # This will run 8 jobs with seeds from 0 to 7

# SEED=$SLURM_ARRAY_TASK_ID  # This will retrieve the current job's array index, which we'll use as the seed

# Declare the datasets and seeds
declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a seeds=("0" "1" "2" "3" "4")


# # Map the SLURM_ARRAY_TASK_ID to a dataset and seed
DATASET=${datasets[$((SLURM_ARRAY_TASK_ID % 3))]}
SEED=${seeds[$((SLURM_ARRAY_TASK_ID / 3))]}


# Since you only have one dataset, you don't need the datasets array
# DATASET="fashionmnist"

# Use SLURM_ARRAY_TASK_ID directly to get the seed
# SEED=$SLURM_ARRAY_TASK_ID


srun bash -c "python experiments/data_experiments.py --seed $SEED --dataset $DATASET"


exit 3