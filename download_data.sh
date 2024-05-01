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
# # LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/topology_experiment_results/jorge_setting_fedavg")
# LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/learning_hive/debug_cifar/heuristic_budget_0_enforce_balance_True_mem_32")
# LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/learning_hive/after_fix_vanilla_cifar_results/heuristic_budget_20_enforce_balance_True_mem_160_sync_base_False_hash_True")
#LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/learning_hive/after_fix_vanilla_cifar_results/heuristic_budget_20_enforce_balance_False_mem_160_sync_base_False_hash_True")
# LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/rerun_topology_experiment_results")
# LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/learning_hive/cifar_heuristic_results/budget")
# LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/learning_hive/budget_and_topology_fedprox_results")
# LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/learning_hive/new_topology_experiment_results/data")
LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/learning_hive/new_topology_experiment_results/data/topology_random_disconnect_edge_drop_1.0")
LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/rerun_cifar_mono_at_more_fl_results")
#LOCAL_FOLDERS=("/mnt/kostas-graid/datasets/vlongle/learning_hive/more_fl_fix_root_agent_results")
REMOTE_USER="vlongle"
REMOTE_HOST="158.130.50.18"
# REMOTE_PATH="/home/vlongle/code/learning-hive/new_topology_experiment_results/"
# REMOTE_PATH="/home/vlongle/code/learning-hive/rerun_opt_heuristic_experiment_results/"
#REMOTE_PATH="/home/vlongle/code/learning-hive/after_fix_vanilla_cifar_results/"
REMOTE_PATH="/home/vlongle/code/learning-hive/new_topology_experiment_results"
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
