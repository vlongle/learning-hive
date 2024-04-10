#!/bin/bash
#SBATCH --output=slurm_outs/modmod/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-8

declare -a topologies=("ring" "tree" "server")
declare -a datasets=("mnist" "kmnist" "fashionmnist")

# Calculate indices for topology and dataset based on SLURM_ARRAY_TASK_ID
TOPOLOGY_IDX=$((SLURM_ARRAY_TASK_ID / 3))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % 3))

# Map the SLURM_ARRAY_TASK_ID to topology and dataset
TOPOLOGY=${topologies[$TOPOLOGY_IDX]}
DATASET=${datasets[$DATASET_IDX]}

# Adjust the command to include topology and dataset, removing the algo parameter
srun bash -c "RAY_DEDUP_LOGS=0 python experiments/modmod_experiments.py --dataset $DATASET --topology $TOPOLOGY --transfer_decoder 1 --transfer_structure 1 --no_sparse_basis 1"

exit 3
