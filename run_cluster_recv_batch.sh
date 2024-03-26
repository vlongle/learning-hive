#!/bin/bash
#SBATCH --output=slurm_outs/recv/slurm-%j.out
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-3 # This will run 8 jobs with seeds from 0 to 7


# Declare the datasets and seeds
declare -a num_queries=("10" "30")
declare -a num_comms_per_task=("1" "10")


NUM_QUERIES=${num_queries[$SLURM_ARRAY_TASK_ID % 2]}
NUM_COMPS_PER_TASK=${num_comms_per_task[$SLURM_ARRAY_TASK_ID / 2]}



srun bash -c "python experiments/recv_experiments.py --num_queries $NUM_QUERIES --num_comms_per_task $NUM_COMPS_PER_TASK"


exit 3