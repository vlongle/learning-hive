#!/bin/bash

# Base directories
src_dir="combined_recv_remove_neighbors_results"
dest_base_dir="budget_experiment_results/jorge_setting_recv_variable_shared_memory_size"

# Find all directories in the source directory that need to be moved
find "$src_dir" -type d -name "combined_modular_numtrain_*" | while read src_subdir; do
  # Extract the relative path of the parent directory of the source subdirectory
  relative_path=$(dirname "${src_subdir#$src_dir/}")

  # Construct the destination directory path
  dest_subdir="$dest_base_dir/$relative_path/$(basename "$src_subdir")"

  # Check if the destination subdirectory exists
  if [ -d "$dest_subdir" ]; then
    echo "Replacing $dest_subdir"
    # Remove the existing destination directory
    rm -rf "$dest_subdir"
  else
    echo "Destination $dest_subdir does not exist, creating it."
  fi

  # Move the source subdirectory to the destination
  mv "$src_subdir" "$(dirname "$dest_subdir")"

  echo "Moved $src_subdir to $dest_subdir"
done
