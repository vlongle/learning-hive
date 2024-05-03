#!/bin/bash
#SBATCH --output=slurm_outs/recv/%A_%a.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=72:00:00
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --array=0-6

# Define combine options
declare -a combine_options=("recv_data+modmod+grad_sharing" "modmod+grad_sharing" "modmod+recv_data" "recv_data+grad_sharing" "modmod" "recv_data" "grad_sharing")

# Get the specific option for the current task
COMBINE=${combine_options[$SLURM_ARRAY_TASK_ID]}

# Execute the Python script with the specific combine option
srun bash -c "python experiments/combine_modes_experiments.py --combine $COMBINE"

exit 3
