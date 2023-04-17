#!/bin/bash
srun --gpus=2\
 --nodes=1\
 --cpus-per-gpu=8\
 --mem-per-cpu=4G\
 --time=72:00:00\
 --exclude=ee-3090-0.grasp.maas\
 --partition=batch \
bash -c "python experiments/experiments_temp.py"

exit 3