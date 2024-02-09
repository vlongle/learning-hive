#!/bin/bash
#SBATCH --output=slurm_outs/modmod/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-5 # This will run 6 jobs, suitable for 3 datasets x 2 algorithms

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a algos=("modular" "monolithic")

# Calculate dataset and algo index
# Each dataset is paired with each algo exactly once
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 2 % 3))
ALGO_INDEX=$((SLURM_ARRAY_TASK_ID % 2))

DATASET=${datasets[$DATASET_INDEX]}
ALGO=${algos[$ALGO_INDEX]}

srun bash -c "python experiments/recv_experiments.py --dataset $DATASET --algo $ALGO"
exit 3
