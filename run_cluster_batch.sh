#!/bin/bash
#SBATCH --output=slurm_outs/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-3   # This will run 4 jobs with seeds from 0 to 3

SEED=$SLURM_ARRAY_TASK_ID  # This will retrieve the current job's array index, which we'll use as the seed

srun bash -c "python experiments/experiments.py --seed $SEED"

exit 3