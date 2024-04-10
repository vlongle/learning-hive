#!/bin/bash
#SBATCH --output=slurm_outs/modmod/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --partition=batch
#SBATCH --exclude=ee-3090-1.grasp.maas
#SBATCH --array=0-2  # Adjusted for 3 topologies

DATASET="combined"
declare -a topologies=("ring" "tree" "server")

TOPOLOGY_IDX=$SLURM_ARRAY_TASK_ID

# Map the SLURM_ARRAY_TASK_ID to topology
TOPOLOGY=${topologies[$TOPOLOGY_IDX]}


srun bash -c "RAY_DEDUP_LOGS=0 python experiments/modmod_experiments.py --topology $TOPOLOGY --transfer_decoder 1 --transfer_structure 1 --no_sparse_basis 1 --dataset $DATASET"

exit 0
