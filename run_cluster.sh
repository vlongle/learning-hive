#!/bin/bash
srun --gpus=4\
 --nodes=1\
 --cpus-per-gpu=8\
 --mem-per-cpu=4G\
 --time=72:00:00 \
bash -c "python experiments/experiments.py"

exit 3