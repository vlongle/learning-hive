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
#SBATCH --array=0-23

# Fixed topology
TOPOLOGY="random"

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a algos=("modular" "monolithic")
declare -a edge_drop_probs=("0.25" "0.5" "0.7" "0.9")

# Calculate indices for dataset, algo, and edge drop probability based on SLURM_ARRAY_TASK_ID
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 8 % 3))
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 4 % 2))
EDGE_DROP_PROB_IDX=$((SLURM_ARRAY_TASK_ID % 4))

# Map the SLURM_ARRAY_TASK_ID to dataset, algo, and edge drop probability
DATASET=${datasets[$DATASET_IDX]}
ALGO=${algos[$ALGO_IDX]}
EDGE_DROP_PROB=${edge_drop_probs[$EDGE_DROP_PROB_IDX]}

# Adjust the command to include dataset, algo, topology, and edge drop probability
srun bash -c "RAY_DEDUP_LOGS=0 python experiments/fedavg_experiments.py --algo $ALGO --dataset $DATASET --topology $TOPOLOGY --comm_freq 5 --edge_drop_prob $EDGE_DROP_PROB"

exit 3