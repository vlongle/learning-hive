#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-7 # For 8 seeds * 2 algos * 2 scale_shared_memory values

# Fixed values
# BUDGET="20000"
BUDGET="0"
NUM_COMPS_PER_TASK="5"
DATASET="cifar100"
ENFORCE_BALANCE="1"

# Declare the seeds, algos, and scale_shared_memory values
declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")
declare -a algos=("modular" "monolithic")
declare -a scale_shared_memories=("0" "1")

# Calculate the index for each option based on SLURM_ARRAY_TASK_ID
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 8 % 2))
SCALE_SHARED_MEMORY_IDX=$((SLURM_ARRAY_TASK_ID / 16 % 2))

SEED=${seeds[$SEED_IDX]}
ALGO=${algos[$ALGO_IDX]}
SCALE_SHARED_MEMORY=${scale_shared_memories[$SCALE_SHARED_MEMORY_IDX]}

# Now include the --scale_shared_memory parameter in your command
srun bash -c "RAY_DEDUP_LOGS=0 python experiments/heuristic_experiments.py --dataset $DATASET --budget $BUDGET --num_comms_per_task $NUM_COMPS_PER_TASK  --seed $SEED --algo $ALGO --enforce_balance $ENFORCE_BALANCE"
# srun bash -c "RAY_DEDUP_LOGS=0 python experiments/recv_experiments.py --dataset $DATASET --num_comms_per_task $NUM_COMPS_PER_TASK --seed $SEED --algo $ALGO --scale_shared_memory $SCALE_SHARED_MEMORY"

exit 3 # Indicate successful completion
