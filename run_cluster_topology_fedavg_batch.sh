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
#SBATCH --array=0-71 # Adjusted for 3 topologies * 3 datasets * 8 seeds

# Declare the arrays for topologies and datasets
declare -a topologies=("ring" "tree" "server")
declare -a datasets=("mnist" "fashionmnist" "kmnist")

# Calculate the indices for topology, dataset, and seed
TOPOLOGY_IDX=$((SLURM_ARRAY_TASK_ID / 24))  # 24 jobs per topology (3 datasets * 8 seeds)
DATASET_IDX=$(((SLURM_ARRAY_TASK_ID / 8) % 3))  # 8 jobs per dataset
SEED=$((SLURM_ARRAY_TASK_ID % 8))  # 8 seeds

# Map the SLURM_ARRAY_TASK_ID to topology and dataset
TOPOLOGY=${topologies[$TOPOLOGY_IDX]}
DATASET=${datasets[$DATASET_IDX]}

# You might need to define or adjust these variables based on your requirements
ALGO="modular"
COMM_FREQ="5"
MU="0.001"

# Run the command with the specified parameters
srun bash -c "python experiments/fedprox_experiments.py --algo $ALGO --comm_freq $COMM_FREQ --seed $SEED --dataset $DATASET --mu $MU --topology $TOPOLOGY"

exit 0
