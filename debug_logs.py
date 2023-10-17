from pprint import pprint


# Open the files and read their contents
with open("fedavg.out", "r") as f:
    fedavg_lines = f.readlines()

with open("fedprox.out", "r") as f:
    fedprox_lines = f.readlines()


# Extracting lines without timestamps
fedavg_cleaned = [line.split("[INFO]")[1] if "[INFO]" in line else line for line in fedavg_lines]
fedprox_cleaned = [line.split("[INFO]")[1] if "[INFO]" in line else line for line in fedprox_lines]

fedprox_cleaned = [line for line in fedprox_cleaned if "aux_loss" not in line]



# Identifying differences
differences_cleaned = [(i, fedavg_cleaned[i], fedprox_cleaned[i]) for i in range(min(len(fedavg_cleaned), len(fedprox_cleaned))) if fedavg_cleaned[i] != fedprox_cleaned[i]]

# Returning the first 20 differences for brevity
pprint(differences_cleaned[:20])
