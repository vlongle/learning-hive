#!/bin/bash
#SBATCH --output=slurm_outs/fl/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-high
#SBATCH --partition=eaton-compute
#SBATCH --array=1-7

# Declare the communication frequencies and datasets
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")  # 8 options

# Fix ALGO to "modular"
ALGO="modular"
DATASET="cifar100"

# Compute seed index
SEED=${seeds[SLURM_ARRAY_TASK_ID]}  # Directly use SLURM_ARRAY_TASK_ID as index

srun bash -c "python experiments/fedavg_experiments.py --algo $ALGO --seed $SEED --dataset $DATASET"

exit 0
