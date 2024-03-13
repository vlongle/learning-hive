#!/bin/bash
#SBATCH --output=slurm_outs/vanilla/%A_%a.out  # Adjusted directory name for clarity
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-7 # Adjusted for 8 jobs, ranging from 0 to 7

# Fixed algorithm as "modular"
ALGO="modular"

# Calculate the seed based on SLURM_ARRAY_TASK_ID (0 to 7)
SEED=$SLURM_ARRAY_TASK_ID

# Use srun with a command to set ulimits inside the compute node before running the Python script
srun bash -c "ulimit -u 100000; ulimit -n 100000; python experiments/experiments.py --seed $SEED --algo $ALGO"

exit 3
