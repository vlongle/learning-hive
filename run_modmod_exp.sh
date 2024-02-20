#!/bin/bash
#SBATCH --output=slurm_outs/modmod/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-23 # This will run 24 jobs

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a sync_base_values=("0" "1") # Use 0 for False, 1 for True
declare -a opt_with_random_values=("0" "1") # Use 0 for False, 1 for True
declare -a freeze_candidate_module_values=("0" "1") # Use 0 for False, 1 for True

# Calculate dataset index
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

# Calculate sync_base value index
SYNC_BASE_INDEX=$(((SLURM_ARRAY_TASK_ID / 3) % 2))

# Calculate opt_with_random value index
OPT_WITH_RANDOM_INDEX=$(((SLURM_ARRAY_TASK_ID / 6) % 2))

# Calculate freeze_candidate_module value index
FREEZE_CANDIDATE_MODULE_INDEX=$(((SLURM_ARRAY_TASK_ID / 12) % 2))

DATASET=${datasets[$DATASET_INDEX]}
SYNC_BASE=${sync_base_values[$SYNC_BASE_INDEX]}
OPT_WITH_RANDOM=${opt_with_random_values[$OPT_WITH_RANDOM_INDEX]}
FREEZE_CANDIDATE_MODULE=${freeze_candidate_module_values[$FREEZE_CANDIDATE_MODULE_INDEX]}

srun bash -c "python experiments/modmod_experiments.py --dataset $DATASET --sync_base $SYNC_BASE --opt_with_random $OPT_WITH_RANDOM --freeze_candidate_module $FREEZE_CANDIDATE_MODULE"
exit 3
