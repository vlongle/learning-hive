#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-5 # 8 jobs for 8 seeds with 1 dataset

# Fixed values
NUM_QUERIES="20"
NUM_COMPS_PER_TASK="5"
ALGO="modular" # Fixed algorithm
DATASET="kmnist" # Fixed dataset

# Declare the seeds
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7") # 8 options

# Calculate the index for seeds based on SLURM_ARRAY_TASK_ID
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))
SEED=${seeds[$SEED_IDX]}

srun bash -c "python experiments/recv_experiments.py --num_queries $NUM_QUERIES --num_comms_per_task $NUM_COMPS_PER_TASK --seed $SEED --algo $ALGO --dataset $DATASET"

exit 0
