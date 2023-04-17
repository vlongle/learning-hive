#!/bin/bash
srun --gpus=1\
 --partition=eaton-compute\
 --qos=ee-med \
bash download_zip_data.sh

exit 3