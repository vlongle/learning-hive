# run 8 seeds 0-7 for --dataset combined

for seed in 0 1 2 3 4 5 6 7
do
    python experiments/heuristic_data_experiments.py --seed $seed --dataset combined &
done