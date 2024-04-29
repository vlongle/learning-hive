#!/bin/bash
#SBATCH --output=slurm_outs/topology/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48  # Keeping the computational resources the same
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=61-61 # Adjusted for 2 algos * 4 edge_drop_probs * 1 dataset * 8 seeds = 64 jobs

# Declare algorithms, edge drop probabilities, and datasets
declare -a algos=("modular" "monolithic")
declare -a edge_drop_probs=("0.25" "0.5" "0.7" "0.9")
declare -a datasets=("combined")  # Only using the combined dataset

# Calculate indices for the algorithm, edge drop probability, dataset, and seed
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 32))  # 32 jobs per algorithm
EDGE_DROP_PROB_IDX=$((SLURM_ARRAY_TASK_ID / 8 % 4))  # 8 jobs per edge drop probability
DATASET_IDX=0  # There is only one dataset, index is always 0
SEED=$((SLURM_ARRAY_TASK_ID % 8))  # 8 seeds

# Map the SLURM_ARRAY_TASK_ID to algorithm, edge drop probability, dataset, and seed
ALGO=${algos[$ALGO_IDX]}
EDGE_DROP_PROB=${edge_drop_probs[$EDGE_DROP_PROB_IDX]}
DATASET=${datasets[$DATASET_IDX]}  # Always "combined"
TOPOLOGY="random_disconnect"

# Execute the experiment with the selected parameters
srun bash -c "python experiments/recv_experiments.py --algo $ALGO --seed $SEED --dataset $DATASET --topology $TOPOLOGY --edge_drop_prob $EDGE_DROP_PROB"

exit 0
