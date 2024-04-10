#!/bin/bash
#SBATCH --output=slurm_outs/modmod/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48  # Assuming you want to keep the computational resources the same
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-7  # Adjusted for 8 jobs in total

# Fixed topology
TOPOLOGY="random_disconnect"
DATASET="combined"

declare -a algos=("modular" "monolithic")
declare -a edge_drop_probs=("0.25" "0.5" "0.7" "0.9")

# Calculate indices for algo and edge drop probability based on SLURM_ARRAY_TASK_ID
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 4 % 2))
EDGE_DROP_PROB_IDX=$((SLURM_ARRAY_TASK_ID % 4))

# Map the SLURM_ARRAY_TASK_ID to algo and edge drop probability
ALGO=${algos[$ALGO_IDX]}
EDGE_DROP_PROB=${edge_drop_probs[$EDGE_DROP_PROB_IDX]}


srun bash -c "RAY_DEDUP_LOGS=0 python experiments/modmod_experiments.py --topology $TOPOLOGY --edge_drop_prob $EDGE_DROP_PROB --transfer_decoder 1 --transfer_structure 1 --no_sparse_basis 1 --dataset $DATASET"

exit 3
