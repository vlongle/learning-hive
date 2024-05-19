#!/bin/bash
#SBATCH --output=slurm_outs/vanilla/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-high
#SBATCH --partition=eaton-compute
#SBATCH --array=0-79 # 80 jobs (2 algos * 1 dataset * 8 seeds * 5 temperatures)

declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7")
declare -a temperatures=("0.1" "0.01" "2.0" "10.0" "100.0")
declare -a datasets=("cifar100")
declare -a algos=("modular" "monolithic")

# Calculate indices for seed, temperature, and algo
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))
TEMP_IDX=$(((SLURM_ARRAY_TASK_ID / 8) % 5))
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 40))

SEED=${seeds[$SEED_IDX]}
TEMP=${temperatures[$TEMP_IDX]}
DATASET=${datasets[0]} # Only one dataset
ALGO=${algos[$ALGO_IDX]}

# Use srun to execute the job with the selected algorithm and any additional settings
srun bash -c "python experiments/fedfish_experiments.py --seed $SEED --temperature $TEMP --dataset $DATASET --algo $ALGO"

exit 0