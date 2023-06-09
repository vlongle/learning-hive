#!/bin/bash

# LOCAL_FOLDER="cifar_no_updates_contrastive_results"
LOCAL_FOLDER="cifar_epochs_500_mild_dropout_memory_64_data_300_results"
REMOTE_USER="vlongle"
REMOTE_HOST="158.130.50.18"
REMOTE_PATH="/home/vlongle/code/learning-hive"

# Check if rsync is installed
if ! command -v rsync &> /dev/null; then
    echo "rsync is not installed. Please install it and try again."
    exit 1
fi

# Upload the folder using rsync over SSH
rsync -avzP --compress-level=9 -e ssh "${LOCAL_FOLDER}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

echo "Folder ${LOCAL_FOLDER} has been successfully uploaded to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"