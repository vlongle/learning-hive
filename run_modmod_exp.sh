#!/bin/bash
#SBATCH --output=slurm_outs/modmod/%A_%a.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-3 # Only 4 jobs in total, combinations of settings

# Calculate settings based on SLURM_ARRAY_TASK_ID
case $SLURM_ARRAY_TASK_ID in
    0)
        TRANSFER_DECODER="0"
        TRANSFER_STRUCTURE="0"
        ;;
    1)
        TRANSFER_DECODER="0"
        TRANSFER_STRUCTURE="1"
        ;;
    2)
        TRANSFER_DECODER="1"
        TRANSFER_STRUCTURE="0"
        ;;
    3)
        TRANSFER_DECODER="1"
        TRANSFER_STRUCTURE="1"
        ;;
esac

NO_SPARSE_BASIS="1"

# Use SLURM_ARRAY_TASK_ID as the seed for each job

# Run the command with the determined settings and a unique seed for each job
srun bash -c "python experiments/modmod_experiments.py --transfer_decoder $TRANSFER_DECODER --transfer_structure $TRANSFER_STRUCTURE --no_sparse_basis $NO_SPARSE_BASIS"
exit 0
