#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-7

# Fixed values
BUDGET="20"  # (20, 40, 80)
NUM_COMPS_PER_TASK="5" # (1,5, 10)
DATASET="cifar100"
ENFORCE_BALANCE="0"

# Declare the seeds and algos
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7") # 8 options
# declare -a algos=("modular" "monolithic") # 2 options
ALGO="monolithic"

# Calculate the index for each option based on SLURM_ARRAY_TASK_ID
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))
# ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 8)) # This division will floor towards 0 for the first 8 jobs and be 1 for the next 8

SEED=${seeds[$SEED_IDX]}
# ALGO=${algos[$ALGO_IDX]}

srun bash -c "RAY_DEDUP_LOGS=0 python experiments/heuristic_data_experiments.py --dataset $DATASET --budget $BUDGET --num_comms_per_task $NUM_COMPS_PER_TASK --seed $SEED --algo $ALGO --enforce_balance $ENFORCE_BALANCE"
# srun bash -c "RAY_DEDUP_LOGS=0 python experiments/recv_experiments.py --dataset $DATASET --num_comms_per_task $NUM_COMPS_PER_TASK --seed $SEED --algo $ALGO"

exit 3