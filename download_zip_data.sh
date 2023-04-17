#!/bin/bash

LOCAL_FOLDER="cifar_no_updates_contrastive_results"
TARBALL_NAME="cifar_no_updates_contrastive_results.tar.gz"
REMOTE_USER="vlongle"
REMOTE_HOST="158.130.50.18"
REMOTE_PATH="/home/vlongle/code/learning-hive"

# Check if tar, scp, and pv are installed
if ! command -v tar &> /dev/null || ! command -v scp &> /dev/null || ! command -v pv &> /dev/null; then
    echo "tar, scp, and/or pv are not installed. Please install them and try again."
    exit 1
fi

# Compress the folder into a tarball
echo "Compressing the folder..."
tar -czf - "${LOCAL_FOLDER}" | pv -s "$(du -sb "${LOCAL_FOLDER}" | awk '{print $1}')" > "${TARBALL_NAME}"

# Upload the tarball using scp
echo "Uploading the tarball..."
pv "${TARBALL_NAME}" | scp -C -o "CompressionLevel=9" - "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${TARBALL_NAME}"

# Extract the tarball on the remote server
echo "Extracting the tarball on the remote server..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "pv ${REMOTE_PATH}/${TARBALL_NAME} | tar -C ${REMOTE_PATH} -xzf - && rm ${REMOTE_PATH}/${TARBALL_NAME}"

# Remove the local tarball
echo "Cleaning up the local tarball..."
rm "${TARBALL_NAME}"

echo "Folder ${LOCAL_FOLDER} has been successfully uploaded to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"