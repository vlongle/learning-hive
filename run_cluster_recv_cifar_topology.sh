#!/bin/bash
#SBATCH --output=slurm_outs/topology/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-47  # Adjusted for 3 topologies * 1 dataset * 8 seeds * 2 algos = 48 jobs

# Declare the arrays for topologies, datasets, and algorithms
declare -a topologies=("ring" "tree" "server")
declare -a datasets=("cifar100")  # Using one dataset
declare -a algos=("modular" "monolithic")

# Calculate the indices for topology, dataset, seed, and algorithm
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 24))  # 24 jobs per algorithm (3 topologies * 1 dataset * 8 seeds)
TOPOLOGY_IDX=$((SLURM_ARRAY_TASK_ID / 8 % 3))  # 8 jobs per topology
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 8 % 1))  # 8 jobs per dataset, % 1 has no effect, just for clarity
SEED=$((SLURM_ARRAY_TASK_ID % 8))  # 8 seeds

# Map the SLURM_ARRAY_TASK_ID to algorithm, topology, and dataset
ALGO=${algos[$ALGO_IDX]}
TOPOLOGY=${topologies[$TOPOLOGY_IDX]}
DATASET=${datasets[$DATASET_IDX]}  # Always "combined"

# Run the command with the specified parameters
srun bash -c "python experiments/heuristic_data_experiments.py --algo $ALGO --seed $SEED --dataset $DATASET --topology $TOPOLOGY"

exit 0
