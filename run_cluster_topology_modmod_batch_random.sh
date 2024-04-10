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
#SBATCH --array=0-11

# Since topology is fixed to 'random', the topologies array and calculation are no longer needed
TOPOLOGY="random"

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a edge_drop_probs=("0.25" "0.5" "0.7" "0.9")

# Adjust the calculation for dataset index and edge drop probability index based on the new script logic
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 4 % 3))
EDGE_DROP_PROB_IDX=$((SLURM_ARRAY_TASK_ID % 4))

# Map the SLURM_ARRAY_TASK_ID to dataset and edge drop probability
DATASET=${datasets[$DATASET_IDX]}
EDGE_DROP_PROB=${edge_drop_probs[$EDGE_DROP_PROB_IDX]}

# Adjust the command to exclude algo and include dataset, topology, and edge drop probability
srun bash -c "RAY_DEDUP_LOGS=0 python experiments/modmod_experiments.py --dataset $DATASET --topology $TOPOLOGY --edge_drop_prob $EDGE_DROP_PROB --transfer_decoder 1 --transfer_structure 1 --no_sparse_basis 1"

exit 3
