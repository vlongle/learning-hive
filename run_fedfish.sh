#!/bin/bash
#SBATCH --output=slurm_outs/vanilla/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-high
#SBATCH --partition=eaton-compute
#SBATCH --array=0-47 # 48 jobs (8 seeds * 3 datasets * 2 algos)

declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")
declare -a datasets=("mnist" "fashionmnist" "kmnist")
declare -a algos=("modular" "monolithic")

# Calculate indices for seed, dataset, and algo
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))
DATASET_IDX=$(((SLURM_ARRAY_TASK_ID / 8) % 3))
ALGO_IDX=$(((SLURM_ARRAY_TASK_ID / 24) % 2))

SEED=${seeds[$SEED_IDX]}
DATASET=${datasets[$DATASET_IDX]}
ALGO=${algos[$ALGO_IDX]}

# Use srun to execute the job with the selected algorithm and any additional settings
srun bash -c "python experiments/fedfish_experiments.py --sync_base true --seed $SEED --dataset $DATASET --algo $ALGO"

exit 0
