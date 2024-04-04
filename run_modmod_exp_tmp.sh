#!/bin/bash
#SBATCH --output=slurm_outs/modmod/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --array=1-8  # Only run 8 tasks for 8 seeds

declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")

# Calculate index for seeds based on SLURM_ARRAY_TASK_ID
seed_index=$(((SLURM_ARRAY_TASK_ID - 1))) # Directly map task ID to seed index

TRANSFER_DECODER="1"
TRANSFER_STRUCTURE="1"
NO_SPARSE_BASIS="1"

SEED=${seeds[$seed_index]}
NUM_SHARED_MODULE="1"  # Fixed at "1"

srun bash -c "RAY_DEDUP_LOGS=0 python experiments/modmod_experiments_tmp.py --transfer_decoder $TRANSFER_DECODER --transfer_structure $TRANSFER_STRUCTURE --num_shared_module $NUM_SHARED_MODULE --no_sparse_basis $NO_SPARSE_BASIS --sync_base true --seed $SEED"
exit 3
