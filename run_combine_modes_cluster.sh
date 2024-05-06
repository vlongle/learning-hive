#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --array=0-20 # Total of 72 tasks (8 seeds * 3 datasets * 3 combine options)

# Define combine options
declare -a combine_options=("recv_data+grad_sharing" "grad_sharing" "recv_data")
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")  # 8 options
declare -a datasets=("mnist" "kmnist" "fashionmnist")

ALGO="monolithic"

# Calculate indices based on SLURM_ARRAY_TASK_ID
task_per_option=$((8 * 3)) # 8 seeds * 3 datasets
combine_index=$(($SLURM_ARRAY_TASK_ID / $task_per_option))
inner_index=$(($SLURM_ARRAY_TASK_ID % $task_per_option))
dataset_index=$(($inner_index / 8))
seed_index=$(($inner_index % 8))

# Get the specific combine option, dataset, and seed for the current task
COMBINE=${combine_options[$combine_index]}
DATASET=${datasets[$dataset_index]}
SEED=${seeds[$seed_index]}

# Execute the Python script with the specific combine option, dataset, and seed
srun bash -c "python experiments/combine_modes_experiments.py --combine $COMBINE --seed $SEED --dataset $DATASET --algo $ALGO"

exit 0