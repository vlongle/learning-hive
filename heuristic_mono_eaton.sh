#!/bin/bash
#SBATCH --output=slurm_outs/cifar_batch_mono_recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=16-50 # For 3*3*8=72 combinations

# Declare budgets, number of computations per task, and seeds
declare -a budgets=("20" "40" "80")  # 3 budget options
declare -a num_comps_per_task=("1" "5" "10")  # 3 options
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")  # 8 options

# Constants
DATASET="cifar100"
ENFORCE_BALANCE="0"
ALGO="monolithic"

# Calculate indices for budgets, num_comps_per_task, and seeds based on SLURM_ARRAY_TASK_ID
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))
NUM_COMPS_PER_TASK_IDX=$(((SLURM_ARRAY_TASK_ID / 8) % 3))
BUDGET_IDX=$((SLURM_ARRAY_TASK_ID / 24))

BUDGET=${budgets[$BUDGET_IDX]}
NUM_COMPS_PER_TASK=${num_comps_per_task[$NUM_COMPS_PER_TASK_IDX]}
SEED=${seeds[$SEED_IDX]}

srun bash -c "python experiments/heuristic_data_experiments.py --dataset $DATASET --budget $BUDGET --num_comms_per_task $NUM_COMPS_PER_TASK --seed $SEED --algo $ALGO --enforce_balance $ENFORCE_BALANCE"

exit 0
