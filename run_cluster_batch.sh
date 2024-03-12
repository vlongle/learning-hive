#!/bin/bash
#SBATCH --output=slurm_outs/vanilla/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-15 # Adjust this for 16 jobs, ranging from 0 to 15

# Declare the algorithms
declare -a algos=("monolithic" "modular")

# Calculate the algorithm index based on SLURM_ARRAY_TASK_ID (0 or 1)
ALGO_INDEX=$((SLURM_ARRAY_TASK_ID / 8))

# Calculate the seed based on SLURM_ARRAY_TASK_ID (0 to 7)
SEED=$((SLURM_ARRAY_TASK_ID % 8))

# Assign the algorithm based on the calculated index
ALGO=${algos[$ALGO_INDEX]}

# Run the command with the seed and algo
srun bash -c "python experiments/experiments.py --seed $SEED --algo $ALGO"

exit 0
