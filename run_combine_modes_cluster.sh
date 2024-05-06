#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-high
#SBATCH --partition=eaton-compute
#SBATCH --array=10-31 # Total of 32 tasks (4 combine options * 8 seeds)

# Define combine options
declare -a combine_options=("recv_data+grad_sharing")
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")  # 8 options
declare -a datasets=("mnist" "kmnist" "fashionmnist")

ALGO="monolithic"

# Calculate combine option and seed indices based on SLURM_ARRAY_TASK_ID
combine_index=$(($SLURM_ARRAY_TASK_ID / 8))
seed_index=$(($SLURM_ARRAY_TASK_ID % 8))

# Get the specific option and seed for the current task
COMBINE=${combine_options[$combine_index]}
SEED=${seeds[$seed_index]}

# Execute the Python script with the specific combine option and seed
srun bash -c "python experiments/combine_modes_experiments.py --combine $COMBINE --seed $SEED --dataset $DATASET --algo $ALGO"

exit 0