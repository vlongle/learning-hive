#!/bin/bash
#SBATCH --output=slurm_outs/fed_curv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-127  # For 4 mu * 8 seeds * 4 datasets = 128 combinations

# Declare mu values and seeds
declare -a mus=("0.001" "0.1" "0.01" "1.0")
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")
declare -a datasets=("mnist" "fashionmnist" "kmnist" "combined")  # 4 dataset options

# Constants
ALGO="modular"

# Calculate indices for mu, seed, and dataset based on SLURM_ARRAY_TASK_ID
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 32))
MU_IDX=$(( (SLURM_ARRAY_TASK_ID % 32) / 8 ))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))

MU=${mus[$MU_IDX]}
SEED=${seeds[$SEED_IDX]}
DATASET=${datasets[$DATASET_IDX]}

srun bash -c "python experiments/fedcurv_experiments.py --mu $MU --dataset $DATASET --seed $SEED --algo $ALGO"

exit 0
