#!/bin/bash
#SBATCH --output=slurm_outs/fl/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48  # Keeping the computational resources the same
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-11  # Adjusted for 1 algo * 4 edge_drop_probs * 3 datasets = 12 jobs

# Fixed topology and algorithm
TOPOLOGY="random_disconnect"
ALGO="modular"  # Fixed to modular algorithm

# Adjust the datasets array to include only the three specified datasets
declare -a edge_drop_probs=("0.25" "0.5" "0.7" "0.9")
declare -a datasets=("fashionmnist" "mnist" "kmnist")  # Removed "combined" dataset

# Adjust the calculation of indices for edge drop probability and dataset based on the new array size
EDGE_DROP_PROB_IDX=$((SLURM_ARRAY_TASK_ID / 3 % 4))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % 3))

# Map the SLURM_ARRAY_TASK_ID to edge drop probability and dataset
EDGE_DROP_PROB=${edge_drop_probs[$EDGE_DROP_PROB_IDX]}
DATASET=${datasets[$DATASET_IDX]}

# Adjust the command to include the fixed algorithm, topology, edge drop probability, and dataset
srun bash -c "RAY_DEDUP_LOGS=0 python experiments/fedavg_experiments.py --algo $ALGO --topology $TOPOLOGY --comm_freq 5 --edge_drop_prob $EDGE_DROP_PROB --dataset $DATASET"

exit 3
