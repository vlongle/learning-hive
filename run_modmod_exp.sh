#!/bin/bash
#SBATCH --output=slurm_outs/modmod/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=42
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-7 # Only 8 jobs in total

# Fixed settings for this run
TRANSFER_DECODER="1"
TRANSFER_STRUCTURE="1"
NO_SPARSE_BASIS="1"

# Use SLURM_ARRAY_TASK_ID as the seed for each job
SEED=$SLURM_ARRAY_TASK_ID

# Run the command with the fixed settings and a unique seed for each job
srun bash -c "python experiments/modmod_experiments.py --transfer_decoder $TRANSFER_DECODER --transfer_structure $TRANSFER_STRUCTURE --no_sparse_basis $NO_SPARSE_BASIS --seed $SEED"
exit 0