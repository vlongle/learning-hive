#!/bin/bash
#SBATCH --output=slurm_outs/fl/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-7 # This will run 8 jobs

# Declare the seeds and algo choices
declare -a comm_freqs=("10" "20" "50" "100")
declare -a algos=("modular" "monolithic")

# Map the SLURM_ARRAY_TASK_ID to a seed and algo
# Ensuring proper array access and logic for distribution across the jobs
ALGO=${algos[$((SLURM_ARRAY_TASK_ID % 2))]}
COMM_FREQ_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
COMM_FREQ=${comm_freqs[$COMM_FREQ_INDEX]}

# Fixed dataset
srun bash -c "python experiments/fedavg_experiments.py --algo $ALGO --comm_freq $COMM_FREQ"
