#!/bin/bash
#SBATCH --output=slurm_outs/fl/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-5  # Adjusted for 3 topologies * 2 algorithms = 6 jobs

declare -a topologies=("ring" "tree" "server")
declare -a algos=("modular" "monolithic")

# Calculate indices for topology and algo based on SLURM_ARRAY_TASK_ID
TOPOLOGY_IDX=$((SLURM_ARRAY_TASK_ID / 2))
ALGO_IDX=$((SLURM_ARRAY_TASK_ID % 2))

# Map the SLURM_ARRAY_TASK_ID to topology and algo
TOPOLOGY=${topologies[$TOPOLOGY_IDX]}
ALGO=${algos[$ALGO_IDX]}

srun bash -c "RAY_DEDUP_LOGS=0 python experiments/fedavg_experiments.py --algo $ALGO --topology $TOPOLOGY --comm_freq 5"

exit 3
