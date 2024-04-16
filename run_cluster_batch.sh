#!/bin/bash
#SBATCH --output=slurm_outs/vanilla/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-5 # Two jobs for two algorithms

declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")


SEED=${seeds[$SLURM_ARRAY_TASK_ID]} # Directly use SLURM_ARRAY_TASK_ID

# Use srun to execute the job with the selected algorithm and any additional settings
srun bash -c "python experiments/experiments.py --sync_base true --seed $SEED"

exit 3
