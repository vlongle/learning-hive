#!/bin/bash
#SBATCH --output=slurm_outs/vanilla/%A_%a.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-1 # Two jobs for two algorithms

# Determine algorithm and settings based on SLURM_ARRAY_TASK_ID
case $SLURM_ARRAY_TASK_ID in
    0)
        ALGO="modular"
        # Add any specific settings for the modular algorithm here
        ;;
    1)
        ALGO="monolithic"
        # Add any specific settings for the monolithic algorithm here
        ;;
esac

# Use srun to execute the job with the selected algorithm and any additional settings
srun bash -c "python experiments/experiments.py --algo $ALGO"

exit 3
