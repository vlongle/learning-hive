#!/bin/bash
#SBATCH --output=slurm_outs/fl/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-15 # This will run 16 jobs

# Declare the seeds and algo choices
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")
declare -a algos=("modular" "monolithic")

# Map the SLURM_ARRAY_TASK_ID to a seed and algo
SEED=${seeds[$((SLURM_ARRAY_TASK_ID % 8))]}
ALGO=${algos[$((SLURM_ARRAY_TASK_ID / 8))]}

# Fixed dataset

srun bash -c "python experiments/fedavg_experiments.py --seed $SEED --algo $ALGO"

exit 3
