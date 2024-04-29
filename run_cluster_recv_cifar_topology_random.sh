#!/bin/bash
#SBATCH --output=slurm_outs/cifar_topology/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48  # Keeping the computational resources the same
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-7  # 1 algo * 1 edge_drop_prob * 2 datasets * 8 seeds = 16 jobs

# Declare algorithms, edge drop probabilities, and datasets
declare -a algos=("monolithic")
declare -a edge_drop_probs=("1.0")
declare -a datasets=("cifar100") 

# Calculate indices for the algorithm, edge drop probability, dataset, and seed
ALGO_IDX=0  # There is only one algorithm, index is always 0
EDGE_DROP_PROB_IDX=0  # There is only one edge drop probability, index is always 0
DATASET_IDX=0
# DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 8))  # 8 jobs per dataset
SEED=$((SLURM_ARRAY_TASK_ID % 8))  # 8 seeds

# Map the SLURM_ARRAY_TASK_ID to algorithm, edge drop probability, dataset, and seed
ALGO=${algos[$ALGO_IDX]}
EDGE_DROP_PROB=${edge_drop_probs[$EDGE_DROP_PROB_IDX]}
DATASET=${datasets[$DATASET_IDX]}
TOPOLOGY="random_disconnect"
SCALE_SHARED_MEMORY="0"
# Execute the experiment with the selected parameters
srun bash -c "python experiments/heuristic_data_experiments.py --algo $ALGO --seed $SEED --dataset $DATASET --topology $TOPOLOGY --edge_drop_prob $EDGE_DROP_PROB --scale_shared_memory $SCALE_SHARED_MEMORY"

exit 0
