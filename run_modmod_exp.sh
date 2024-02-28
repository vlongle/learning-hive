#!/bin/bash
#SBATCH --output=slurm_outs/modmod/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-11 # Now 12 jobs in total, for each combination of settings

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a transfer_decoder=("0" "1")
declare -a transfer_structure=("0" "1")

# Calculate dataset index
dataset_index=$(($SLURM_ARRAY_TASK_ID / 4)) # 4 combinations of transfer settings per dataset
# Calculate transfer_decoder index
transfer_decoder_index=$((($SLURM_ARRAY_TASK_ID / 2) % 2)) # Alternates every 2 jobs
# Calculate transfer_structure index
transfer_structure_index=$(($SLURM_ARRAY_TASK_ID % 2)) # Alternates every job

DATASET=${datasets[$dataset_index]}
TRANSFER_DECODER=${transfer_decoder[$transfer_decoder_index]}
TRANSFER_STRUCTURE=${transfer_structure[$transfer_structure_index]}

srun bash -c "python experiments/modmod_experiments.py --dataset $DATASET --transfer_decoder $TRANSFER_DECODER --transfer_structure $TRANSFER_STRUCTURE"
exit 0