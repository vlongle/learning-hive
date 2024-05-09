#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-high
#SBATCH --partition=eaton-compute
#SBATCH --array=0-15 # Total of 24 tasks (8 seeds * 1 dataset * 2 combine options)

# Define combine options
declare -a combine_options=("heuristic_data+grad_sharing" "heuristic_data")
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")  # 8 options
declare -a datasets=("cifar100")  # Only one dataset

ALGO="monolithic"

# Calculate indices based on SLURM_ARRAY_TASK_ID
task_per_option=$((8 * 1)) # 8 seeds * 1 dataset
combine_index=$(($SLURM_ARRAY_TASK_ID / $task_per_option))
inner_index=$(($SLURM_ARRAY_TASK_ID % $task_per_option))
dataset_index=0  # Only one dataset, index is always 0
seed_index=$(($inner_index % 8))

# Get the specific combine option, dataset, and seed for the current task
COMBINE=${combine_options[$combine_index]}
DATASET=${datasets[$dataset_index]}
SEED=${seeds[$seed_index]}

# Execute the Python script with the specific combine option, dataset, and seed
srun bash -c "python experiments/combine_modes_experiments.py --combine $COMBINE --seed $SEED --dataset $DATASET --algo $ALGO"

exit 0
