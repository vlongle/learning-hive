#!/bin/bash
#SBATCH --output=slurm_outs/topology/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-143  # Adjusted for 3 topologies * 3 datasets * 8 seeds * 2 algos = 144 jobs

# Declare the arrays for topologies, datasets, and algorithms
declare -a topologies=("ring" "tree" "server")
declare -a datasets=("fashionmnist" "mnist" "kmnist")
declare -a algos=("modular" "monolithic")

# Calculate the indices for topology, dataset, seed, and algorithm
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 72))  # 72 jobs per algorithm (3 topologies * 3 datasets * 8 seeds)
TOPOLOGY_IDX=$((SLURM_ARRAY_TASK_ID / 24 % 3))  # 24 jobs per topology
DATASET_IDX=$(((SLURM_ARRAY_TASK_ID / 8) % 3))  # 8 jobs per dataset
SEED=$((SLURM_ARRAY_TASK_ID % 8))  # 8 seeds

# Map the SLURM_ARRAY_TASK_ID to algorithm, topology, and dataset
ALGO=${algos[$ALGO_IDX]}
TOPOLOGY=${topologies[$TOPOLOGY_IDX]}
DATASET=${datasets[$DATASET_IDX]}

# Run the command with the specified parameters
srun bash -c "python experiments/recv_experiments.py --algo $ALGO --seed $SEED --dataset $DATASET --topology $TOPOLOGY"

exit 0
