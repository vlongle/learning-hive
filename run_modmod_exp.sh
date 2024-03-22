#!/bin/bash
#SBATCH --output=slurm_outs/modmod/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-7 # 8 jobs in total for each combination of settings

declare -a transfer_decoder=("0" "1")
declare -a transfer_structure=("0" "1")
declare -a sync_base=("0" "1")

# Calculate indices for each setting based on the SLURM_ARRAY_TASK_ID
transfer_decoder_index=$((SLURM_ARRAY_TASK_ID / 4 % 2))
transfer_structure_index=$((SLURM_ARRAY_TASK_ID / 2 % 2))
sync_base_index=$((SLURM_ARRAY_TASK_ID % 2))

TRANSFER_DECODER=${transfer_decoder[$transfer_decoder_index]}
TRANSFER_STRUCTURE=${transfer_structure[$transfer_structure_index]}
SYNC_BASE=${sync_base[$sync_base_index]}
NO_SPARSE_BASIS="1" # Statically set to "1"

srun bash -c "python experiments/modmod_experiments.py --transfer_decoder $TRANSFER_DECODER --transfer_structure $TRANSFER_STRUCTURE --no_sparse_basis $NO_SPARSE_BASIS --sync_base $SYNC_BASE"
exit 3
