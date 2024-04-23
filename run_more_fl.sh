#!/bin/bash
#SBATCH --output=slurm_outs/fed_prox/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-31  # For 4*8=32 combinations

# Declare mu values and seeds
declare -a mus=("0.001" "0.1" "0.01" "1.0")
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")  # 8 options

# Constants
DATASET="cifar100"
ALGO="modular"

# Calculate indices for mu and seeds based on SLURM_ARRAY_TASK_ID
MU_IDX=$((SLURM_ARRAY_TASK_ID / 8))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))

MU=${mus[$MU_IDX]}
SEED=${seeds[$SEED_IDX]}

# srun bash -c "python experiments/fedprox_experiments.py --mu $MU --dataset $DATASET --seed $SEED --algo $ALGO"
srun bash -c "python experiments/fedcurv_experiments.py --mu $MU --dataset $DATASET --seed $SEED --algo $ALGO"

exit 3
