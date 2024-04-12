#!/bin/bash


# # Base path for the experiments
# BASE_PATH="experiment_results/jorge_setting_lowest_task_id_wins_modmod_test_sync_base_False_opt_with_random_False_frozen_False"

# # Possible states for the variables
# states=("True" "False")

# # Generate folder names based on combinations of no_sparse_basis, transfer_decoder, and transfer_structure
# LOCAL_FOLDERS=()
# for no_sparse_basis in "${states[@]}"; do
#     for transfer_decoder in "${states[@]}"; do
#         for transfer_structure in "${states[@]}"; do
#             folder="${BASE_PATH}_transfer_decoder_${transfer_decoder}_transfer_structure_${transfer_structure}_no_sparse_basis_${no_sparse_basis}"
#             LOCAL_FOLDERS+=("$folder")
#         done
#     done
# done


# # Array of local folders to download
# LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/topology_experiment_results/modmod")
# LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/topology_experiment_results/jorge_setting_fedavg")
LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/learning_hive/debug_cifar_data/heuristic_budget_0_enforce_balance_True_mem_32")
REMOTE_USER="vlongle"
REMOTE_HOST="158.130.50.18"
# REMOTE_PATH="/home/vlongle/code/learning-hive/new_topology_experiment_results/"
# REMOTE_PATH="/home/vlongle/code/learning-hive/rerun_opt_heuristic_experiment_results/"
REMOTE_PATH="/home/vlongle/code/learning-hive/"
# REMOTE_PATH="/home/vlongle/code/learning-hive/combined_data_experiment_results/"

# Check if rsync is installed
if ! command -v rsync &> /dev/null; then
    echo "rsync is not installed. Please install it and try again."
    exit 1
fi

# Download each folder using rsync over SSH in parallel
for LOCAL_FOLDER in "${LOCAL_FOLDERS[@]}"; do
    rsync -avzP --compress-level=9 -e ssh "${LOCAL_FOLDER}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" &
done

# Wait for all background jobs to finish
wait

echo "All folders have been successfully downloaded from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
