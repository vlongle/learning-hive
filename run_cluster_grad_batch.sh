#!/bin/bash
#SBATCH --output=slurm_outs/fl/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-7 # This will run 8 jobs, iterating over 2 algorithms and 4 communication frequencies

# Declare the seeds and algo choices
declare -a comm_freqs=("10" "20" "50" "100")
declare -a algos=("modular" "monolithic")

# Map the SLURM_ARRAY_TASK_ID to a comm_freq and algo directly
ALGO=${algos[$((SLURM_ARRAY_TASK_ID % 2))]} # Use modulo by the number of algorithms to cycle through algos
COMM_FREQ=${comm_freqs[$((SLURM_ARRAY_TASK_ID / 2))]} # Use integer division by the number of algorithms to cycle through comm_freqs

# Fixed dataset

srun bash -c "python experiments/fedavg_experiments.py --algo $ALGO --comm_freq $COMM_FREQ"

exit 3
