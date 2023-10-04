import shutil
import os

# Set the paths for the two folders
vanilla_results_path = "experiment_results/vanilla_results"
vanilla_more_seeds_results_path = "experiment_results/vanilla_more_trials"

# Get a list of all the job names in the vanilla_more_seeds_results folder
job_names = os.listdir(vanilla_more_seeds_results_path)

# Loop through each job name in the vanilla_more_seeds_results folder
for job_name in job_names:
    if job_name == ".DS_Store":
        continue
    job_name_path = os.path.join(vanilla_more_seeds_results_path, job_name)
    datasets = os.listdir(job_name_path)

    # Loop through each dataset for the current job name
    for dataset in datasets:
        dataset_path = os.path.join(job_name_path, dataset)
        algos = os.listdir(dataset_path)

        # Loop through each algorithm for the current dataset
        for algo in algos:
            algo_path = os.path.join(dataset_path, algo)
            seeds = os.listdir(algo_path)

            # Loop through each seed for the current algorithm
            for seed in seeds:
                seed_path = os.path.join(algo_path, seed)
                vanilla_result_path = os.path.join(
                    vanilla_results_path, job_name, dataset, algo, seed)

                print(seed_path, '\t', vanilla_result_path)
                # Create the necessary directories in the vanilla_results folder
                # os.makedirs(vanilla_result_path, exist_ok=True)

                # Copy the contents of the seed folder to the vanilla_results folder
                shutil.copytree(seed_path, vanilla_result_path)
