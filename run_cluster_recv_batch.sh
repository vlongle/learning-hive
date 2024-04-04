#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-17 # This will run 18 jobs to cover 2 * 3 * 3 combinations

# Declare the datasets and seeds
declare -a num_queries=("10" "20" "30") # 3 options
declare -a num_comms_per_task=("1" "5" "10") # 3 options
declare -a algos=("modular" "monolithic") # 2 options

# Calculate the index for each option based on SLURM_ARRAY_TASK_ID
NUM_QUERIES_IDX=$((SLURM_ARRAY_TASK_ID % 3))
NUM_COMPS_PER_TASK_IDX=$(((SLURM_ARRAY_TASK_ID / 3) % 3))
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 9)) # This division will floor towards 0 for the first 9 jobs and be 1 for the last 9

NUM_QUERIES=${num_queries[$NUM_QUERIES_IDX]}
NUM_COMPS_PER_TASK=${num_comms_per_task[$NUM_COMPS_PER_TASK_IDX]}
ALGO=${algos[$ALGO_IDX]}

srun bash -c "RAY_DEDUP_LOGS=0 python experiments/recv_experiments.py --num_queries $NUM_QUERIES --num_comms_per_task $NUM_COMPS_PER_TASK --algo $ALGO"

exit 3
