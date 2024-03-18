#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=46
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-5 # This will run 48 jobs, suitable for 3 datasets x 2 algorithms x 8 seeds

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a algos=("modular" "monolithic")
# Number of seeds

DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 2 % 3))
ALGO_INDEX=$((SLURM_ARRAY_TASK_ID % 2))


DATASET=${datasets[$DATASET_INDEX]}
ALGO=${algos[$ALGO_INDEX]}

srun bash -c "python experiments/recv_experiments.py --dataset $DATASET --algo $ALGO"
exit 0
