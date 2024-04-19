#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --partition=batch
#SBATCH --array=0-8 # This will run 9 jobs to cover 3 * 3 combinations

# Declare the datasets and seeds
ALGO="monolithic"
DATASET="combined"
declare -a num_queries=("10" "20" "30") # 3 options
declare -a num_comms_per_task=("1" "5" "10") # 3 options

# Calculate the index for each option based on SLURM_ARRAY_TASK_ID
NUM_QUERIES_IDX=$((SLURM_ARRAY_TASK_ID % 3))
NUM_COMPS_PER_TASK_IDX=$(((SLURM_ARRAY_TASK_ID / 3) % 3))

NUM_QUERIES=${num_queries[$NUM_QUERIES_IDX]}
NUM_COMPS_PER_TASK=${num_comms_per_task[$NUM_COMPS_PER_TASK_IDX]}

srun bash -c "python experiments/recv_experiments.py --dataset $DATASET --num_queries $NUM_QUERIES --num_comms_per_task $NUM_COMPS_PER_TASK --algo $ALGO"

exit 3
