#!/bin/bash
#SBATCH --output=slurm_outs/modmod/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-47 # This will run 48 jobs, suitable for 3 datasets x 2 algorithms x 8 seeds

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a algos=("modular" "monolithic")
# Number of seeds
num_seeds=8

# Calculate dataset, algo index, and seed
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / (2 * num_seeds) % 3))
ALGO_INDEX=$((SLURM_ARRAY_TASK_ID / num_seeds % 2))
SEED=$((SLURM_ARRAY_TASK_ID % num_seeds))

DATASET=${datasets[$DATASET_INDEX]}
ALGO=${algos[$ALGO_INDEX]}

srun bash -c "python experiments/recv_experiments.py --dataset $DATASET --algo $ALGO --seed $SEED"
exit 0
