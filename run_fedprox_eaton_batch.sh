#!/bin/bash
#SBATCH --output=slurm_outs/fed_prox/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-14 # This will run 15 jobs

declare -a datasets=("mnist" "kmnist" "fashionmnist")
declare -a seeds=("0" "1")
declare -a mus=("0.0" "0.1" "0.5" "1.0" "2.0")

DATASET=${datasets[$((SLURM_ARRAY_TASK_ID % 3))]}
SEED=${seeds[$((SLURM_ARRAY_TASK_ID / 3 % 2))]}
MU=${mus[$((SLURM_ARRAY_TASK_ID / 6))]}

srun bash -c "python experiments/grad_experiments.py --seed $SEED --dataset $DATASET --mu $MU"
exit 3