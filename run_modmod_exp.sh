#!/bin/bash
#SBATCH --output=slurm_outs/fed_prox/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-11 # This will run 12 jobs

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a sync_base_values=("True" "False")
declare -a opt_with_random_values=("True" "False")

# Calculate dataset index
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
DATASET=${datasets[$DATASET_INDEX]}

# Calculate sync_base value
SYNC_BASE_INDEX=$((SLURM_ARRAY_TASK_ID / 3 % 2))
SYNC_BASE=${sync_base_values[$SYNC_BASE_INDEX]}

# Calculate opt_with_random value
OPT_WITH_RANDOM_INDEX=$((SLURM_ARRAY_TASK_ID / 6 % 2))
OPT_WITH_RANDOM=${opt_with_random_values[$OPT_WITH_RANDOM_INDEX]}

srun bash -c "python experiments/modmod_experiments.py --dataset $DATASET --sync_base $SYNC_BASE --opt_with_random $OPT_WITH_RANDOM"
exit 3