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
#SBATCH --array=0-3  # Adjusted for 2 algos * 2 edge_drop_probs = 4 jobs

# Fixed topology and dataset since now only one dataset is involved
TOPOLOGY="random_disconnect"
DATASET="cifar100"

# Define the algos and adjust the edge_drop_probs array for "0.7" and "0.9" only
declare -a algos=("modular" "monolithic")
declare -a edge_drop_probs=("0.7" "0.9")

# Calculate indices for algo and edge drop probability based on SLURM_ARRAY_TASK_ID
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 2 % 2))
EDGE_DROP_PROB_IDX=$((SLURM_ARRAY_TASK_ID % 2))

# Map the SLURM_ARRAY_TASK_ID to algo and edge drop probability
ALGO=${algos[$ALGO_IDX]}
EDGE_DROP_PROB=${edge_drop_probs[$EDGE_DROP_PROB_IDX]}

# Adjust the command to include the fixed dataset, algo, topology, edge drop probability
srun bash -c "RAY_DEDUP_LOGS=0 python experiments/fedavg_experiments.py --algo $ALGO --topology $TOPOLOGY --comm_freq 5 --edge_drop_prob $EDGE_DROP_PROB --dataset $DATASET"

exit 0
