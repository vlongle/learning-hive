#!/bin/bash
#SBATCH --output=slurm_outs/fl/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-17

declare -a topologies=("ring" "tree" "server")
declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a algos=("modular" "monolithic")

# Calculate indices for topology, dataset, and algo based on SLURM_ARRAY_TASK_ID
TOPOLOGY_IDX=$((SLURM_ARRAY_TASK_ID / 6))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 2 % 3))
ALGO_IDX=$((SLURM_ARRAY_TASK_ID % 2))

# Map the SLURM_ARRAY_TASK_ID to topology, dataset, and algo
TOPOLOGY=${topologies[$TOPOLOGY_IDX]}
DATASET=${datasets[$DATASET_IDX]}
ALGO=${algos[$ALGO_IDX]}

# Adjust the command to include topology and dataset
srun bash -c "RAY_DEDUP_LOGS=0 python experiments/fedavg_experiments.py --algo $ALGO --dataset $DATASET --topology $TOPOLOGY --comm_freq 5"

exit 3