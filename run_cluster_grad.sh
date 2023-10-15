#!/bin/bash
srun --gpus=1\
 --nodes=1\
 --cpus-per-gpu=2\
 --mem-per-cpu=4G\
 --time=1:00:00\
 --qos=ee-med\
 --partition=eaton-compute \
 -w ee-3090-0.grasp.maas \
bash -c "python3 experiments/grad_experiments.py"

exit 3