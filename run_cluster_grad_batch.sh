#!/bin/bash
#SBATCH --output=slurm_outs/fl/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=21-21 # This will run 64 jobs for 4 comm_freqs * 8 seeds * 2 datasets

# Declare the communication frequencies and datasets
declare -a comm_freqs=("10" "20" "50" "100")
declare -a datasets=("combined" "cifar100")

# Fix ALGO to "modular"
ALGO="modular"
MU="0.001"

# Calculate indexes for communication frequency, seed, and dataset
COMM_FREQ_INDEX=$((SLURM_ARRAY_TASK_ID / 16)) # There are 16 jobs for each comm frequency (8 seeds * 2 datasets)
COMM_FREQ=${comm_freqs[$COMM_FREQ_INDEX]}

SEED=$(( (SLURM_ARRAY_TASK_ID / 2) % 8 )) # There are 2 jobs for each seed (different datasets)

DATASET_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
DATASET=${datasets[$DATASET_INDEX]}

# Execute the command with the specified parameters
# srun bash -c "python experiments/fedavg_experiments.py --algo $ALGO --comm_freq $COMM_FREQ --seed $SEED --dataset $DATASET"
srun bash -c "python experiments/fedprox_experiments.py --algo $ALGO --comm_freq $COMM_FREQ --seed $SEED --dataset $DATASET --mu $MU"

exit 0
