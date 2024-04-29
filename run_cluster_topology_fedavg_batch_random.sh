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
#SBATCH --array=0-95  # 1 algo * 4 edge_drop_probs * 3 datasets * 8 seeds = 96 jobs

# Fixed topology and algorithm
TOPOLOGY="random_disconnect"
ALGO="modular"
COMM_FREQ="5"
MU="0.001"

# Define the new datasets and seeds arrays
declare -a edge_drop_probs=("0.25" "0.5" "0.7" "0.9")
declare -a datasets=("mnist" "fashionmnist" "kmnist")
declare -a seeds=("1" "2" "3" "4" "5" "6" "7" "8")

# Calculate the indices for edge drop probability, dataset, and seed
EDGE_DROP_PROB_IDX=$((SLURM_ARRAY_TASK_ID / 24 % 4))  # 24 jobs per edge drop probability (3 datasets * 8 seeds)
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 8 % 3))  # 8 jobs per dataset
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))  # 8 seeds

# Map the SLURM_ARRAY_TASK_ID to edge drop probability, dataset, and seed
EDGE_DROP_PROB=${edge_drop_probs[$EDGE_DROP_PROB_IDX]}
DATASET=${datasets[$DATASET_IDX]}
SEED=${seeds[$SEED_IDX]}

# Run the command with the specified parameters
srun bash -c "python experiments/fedprox_experiments.py --algo $ALGO --comm_freq $COMM_FREQ --seed $SEED --dataset $DATASET --mu $MU --topology $TOPOLOGY --edge_drop_prob $EDGE_DROP_PROB"

exit 0
