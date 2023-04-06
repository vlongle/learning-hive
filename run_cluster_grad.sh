#!/bin/bash
srun --gpus=1\
 --nodes=1\
 --cpus-per-gpu=1\
 --mem-per-cpu=4G\
 --time=72:00:00\
 --qos=ee-med\
 --partition=eaton-compute \
bash -c "python3 experiments/grad_experiments.py"

exit 3