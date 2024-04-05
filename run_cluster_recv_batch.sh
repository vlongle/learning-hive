#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=3-3 # This will run 16 jobs to cover 8 * 2 combinations

# Fixed values
NUM_QUERIES="20"
NUM_COMPS_PER_TASK="5"
SYNC_BASE="1"

# Declare the seeds and algos
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7") # 8 options
declare -a algos=("modular" "monolithic") # 2 options

# Calculate the index for each option based on SLURM_ARRAY_TASK_ID
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 8)) # This division will floor towards 0 for the first 8 jobs and be 1 for the next 8

SEED=${seeds[$SEED_IDX]}
ALGO=${algos[$ALGO_IDX]}

srun bash -c "RAY_DEDUP_LOGS=0 python experiments/recv_experiments.py --num_queries $NUM_QUERIES --num_comms_per_task $NUM_COMPS_PER_TASK --sync_base $SYNC_BASE --seed $SEED --algo $ALGO"

exit 3