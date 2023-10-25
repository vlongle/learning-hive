#!/bin/bash
#SBATCH --output=slurm_outs/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --nodelist=ee-a6000-1.grasp.maas   # Use only the specific node with A6000 GPUs
#SBATCH --array=0-7   # This will run 8 jobs with seeds from 0 to 7

# Use SLURM_ARRAY_TASK_ID directly to get the seed
SEED=$SLURM_ARRAY_TASK_ID

# Since there's only one dataset to consider, directly assign its value
DATASET="cifar100"

srun bash -c "python experiments/experiments.py --seed $SEED --dataset $DATASET"

exit 3