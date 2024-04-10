#!/bin/bash
#SBATCH --output=slurm_outs/fl/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48  # Assuming you want to keep the computational resources the same
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-31  # Adjusted for 2 algos * 4 edge_drop_probs * 4 datasets = 32 jobs

# Fixed topology
TOPOLOGY="random_disconnect"

# Extend the script to handle multiple datasets
declare -a algos=("modular" "monolithic")
declare -a edge_drop_probs=("0.25" "0.5" "0.7" "0.9")
declare -a datasets=("fashionmnist" "mnist" "kmnist" "combined")

# Calculate indices for algo, edge drop probability, and dataset based on SLURM_ARRAY_TASK_ID
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 16 % 2))
EDGE_DROP_PROB_IDX=$((SLURM_ARRAY_TASK_ID / 4 % 4))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % 4))

# Map the SLURM_ARRAY_TASK_ID to algo, edge drop probability, and dataset
ALGO=${algos[$ALGO_IDX]}
EDGE_DROP_PROB=${edge_drop_probs[$EDGE_DROP_PROB_IDX]}
DATASET=${datasets[$DATASET_IDX]}

# Adjust the command to include the fixed dataset, algo, topology, edge drop probability, and dataset
srun bash -c "RAY_DEDUP_LOGS=0 python experiments/fedavg_experiments.py --algo $ALGO --topology $TOPOLOGY --comm_freq 5 --edge_drop_prob $EDGE_DROP_PROB --dataset $DATASET"

exit 3
