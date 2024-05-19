from IPython.display import display, HTML
import omegaconf
from shell.utils.experiment_utils import *
from shell.fleet.utils.fleet_utils import *
from shell.utils.metric import *
import matplotlib.pyplot as plt
from shell.fleet.network import TopologyGenerator
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from shell.fleet.fleet import Agent, Fleet
from shell.fleet.data.data_utilize import *
from shell.fleet.data.recv import *

from sklearn.manifold import TSNE
from torchvision.utils import make_grid
from shell.fleet.data.data_utilize import *
import logging
from sklearn.metrics import f1_score
import os
from shell.fleet.data.recv_utils import *
from pythresh.thresholds.aucp import AUCP
from pythresh.thresholds.boot import BOOT
from pythresh.thresholds.zscore import ZSCORE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import seaborn as sns
from prettytable import PrettyTable

plt.style.use('seaborn-whitegrid')
logging.basicConfig(level=logging.CRITICAL)


def plot_agg_learning_curves(fleet, ax=None, name=None, tasks=None, agent_ids=None, viz=True, mode="current",
                             metric="test_acc", error_type='stderr'):
    if ax is None and viz:
        fig, ax = plt.subplots()

    if tasks is None:
        tasks = range(fleet.num_init_tasks, fleet.num_tasks)
    if tasks == "last":
        tasks = [fleet.num_tasks - 1]

    dfs = []
    for agent in fleet.agents:
        if agent_ids is not None and agent.node_id not in agent_ids:
            continue
        df = agent.get_record().df
        for task in tasks:
            task_df = df[df["train_task"] == task]
            if mode == "current":
                task_df = task_df[task_df["test_task"] == str(task)]
            elif mode == "avg":
                task_df = task_df[task_df["test_task"] == 'avg']
            else:
                raise ValueError("mode must be current or avg")
            task_df['agent_id'] = agent.node_id
            dfs.append(task_df)

    if len(dfs) == 0:
        print("ERR at ", fleet.save_dir)
    combined_df = pd.concat(dfs)

    # Calculate mean, standard deviation, and count (for standard error) grouped by epoch
    agg_df = combined_df.groupby(['epoch']).agg(
        {metric: ['mean', 'std', 'count']}).reset_index()
    # Simplify column names
    agg_df.columns = ['epoch', f'{metric}_mean', f'{metric}_std', 'count']

    # Extract mean and standard deviation values
    mean_test_acc = agg_df[f'{metric}_mean']
    std_test_acc = agg_df[f'{metric}_std']
    count_test_acc = agg_df['count']

    # Calculate standard error if requested
    if error_type == 'stderr':
        error = std_test_acc / np.sqrt(count_test_acc)
    else:
        error = std_test_acc

    if viz:
        # Plot the mean test_acc with shaded areas for standard deviation or standard error
        ax.plot(agg_df['epoch'], mean_test_acc, label=name)
        ax.fill_between(agg_df['epoch'], mean_test_acc -
                        error, mean_test_acc + error, alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        if name is not None:
            ax.legend()

    # Compute area under the curve
    auc = np.trapz(mean_test_acc, agg_df['epoch'])
    return auc, agg_df


class FakeRecord:
    def __init__(self, df):
        self.df = df


class FakeAgent:
    def __init__(self, save_dir, node_id):
        self.node_id = node_id
        self.save_dir = save_dir

    def get_record(self):
        return FakeRecord(pd.read_csv(os.path.join(self.save_dir, "record.csv")))


class FakeFleet:
    def __init__(self, save_dir, num_init_tasks=4):
        self.save_dir = save_dir
        self.num_init_tasks = num_init_tasks
        self.num_tasks = 20 if "cifar100" in save_dir else 10
        self.agents = [FakeAgent(os.path.join(save_dir, agent_id), agent_id) for agent_id in os.listdir(
            save_dir) if agent_id != "hydra_out" and agent_id != "agent_69420"]

    def load_records(self):
        pass


def setup_fake_fleet(result_dir, modify_cfg=None, parallel=False):
    return FakeFleet(result_dir)


def load_data(get_save_dirs, seeds, datasets, modify_cfg, viz, tasks, agent_ids, ax=None, mode="current",
              metric="test_acc", strict=False, **kwargs):
    dataset_seed_aucs = {}
    dataset_agg_dfs = {}
    for dataset in datasets:
        seed_aucs = {}
        agg_dfs = []
        for seed in seeds:
            fleets = {}
            save_dirs = get_save_dirs(dataset, seed)
            for name, save_dir in save_dirs.items():
                try:
                    fleet = setup_fake_fleet(
                        save_dir, modify_cfg=modify_cfg, parallel=False)
                    fleet.load_records()
                    fleets[name] = fleet
                except:
                    if strict:
                        raise ValueError(
                            f"Failed to load {name} for {dataset} seed {seed} @ save_dir {save_dir}")
                    continue

            if viz:
                fig, ax = plt.subplots()

            aucs = {}
            for title_name, fleet in fleets.items():
                aucs[title_name], agg_df = plot_agg_learning_curves(fleet, ax, name=title_name, tasks=tasks, agent_ids=agent_ids, viz=viz, mode=mode,
                                                                    metric=metric, **kwargs)
                agg_df['seed'] = seed
                agg_df['name'] = title_name
                agg_dfs.append(agg_df)
            combined_agg_df = pd.concat(agg_dfs)
            seed_aucs[seed] = aucs

        dataset_seed_aucs[dataset] = seed_aucs
        dataset_agg_dfs[dataset] = combined_agg_df

    return dataset_seed_aucs, dataset_agg_dfs


def plot_agg_over_seeds(combined_agg_df, title_name=None, ax=None, std_scale=1.0, metric='test_acc',
                        remap_name=None, colormap=None, error_type='stderr'):
    if title_name is None:
        title_name = 'Aggregated Test Accuracy Learning Curves Across All Seeds and Algorithms'
    if ax is None:
        fig, ax = plt.subplots()

    # Assuming your DataFrame has columns like 'metric_mean' and 'metric_std' after groupby and aggregation
    metric_mean = f'{metric}_mean'
    metric_std = f'{metric}_std'

    agg_over_seed_name = combined_agg_df.groupby(['name', 'epoch']).agg({
        metric_mean: 'mean',
        metric_std: 'std' if error_type == 'std' else 'sem'
    }).reset_index()

    for name, group in agg_over_seed_name.groupby('name'):
        if remap_name is not None and name not in remap_name:
            continue

        name_display = remap_name[name] if remap_name is not None else name
        mean_values = group[metric_mean]
        std_values = group[metric_std]
        ax.plot(group['epoch'], mean_values, label=name_display, marker='o',
                color=colormap[name_display] if colormap is not None else None)
        ax.fill_between(group['epoch'], mean_values - std_scale * std_values, mean_values + std_scale * std_values, alpha=0.3,
                        color=colormap[name_display] if colormap is not None else None)

    # ax.set_xlabel('Epoch', fontsize=14)
    # ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title(title_name, fontsize=30, weight='bold')
    # Setting legend font size
    # ax.legend(frameon=True, loc='lower right', fontsize=12)
    # Setting x and y ticks font size
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)

    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    # ax.set_ylim(0.5, 0.8)


def get_auc_stats(seed_aucs):
    algo_stats = {}
    for seed, auc in seed_aucs.items():
        for algo, auc_ in auc.items():
            if algo not in algo_stats:
                algo_stats[algo] = {'auc_scores': []}
            algo_stats[algo]['auc_scores'].append(auc_)

    for algo, stats in algo_stats.items():
        scores = stats['auc_scores']
        stats['average_auc'] = np.mean(scores)
        stats['std_auc'] = np.std(scores)
        stats['stderr_auc'] = stats['std_auc'] / np.sqrt(len(scores))
        print(
            f"{algo}: Average AUC = {stats['average_auc']:.2f}, STD = {stats['std_auc']:.2f}, STDERR = {stats['stderr_auc']:.2f}")
    return algo_stats


def plot_auc_combined(dataset_seed_aucs, remap_name=None, colormap=None, mode='avg', error_type='std',
                      save_fig_path=None, bar_width=0.1, figsize=(15, 5),
                      custom_algo_order=None,
                      plot_prefix_name="",
                      min_y=50, max_y=None):
    fig, ax = plt.subplots(figsize=figsize)
    # Initialize variables for plotting
    algo_stats_global = {}

    for dataset, seed_aucs in dataset_seed_aucs.items():
        algo_stats = get_auc_stats(seed_aucs)
        for algo, stats in algo_stats.items():
            if algo not in algo_stats_global:
                algo_stats_global[algo] = {'average_aucs': [], 'errors': []}
            algo_stats_global[algo]['average_aucs'].append(
                stats['average_auc'])
            error_value = stats['std_auc'] if error_type == 'std' else stats['stderr_auc']
            algo_stats_global[algo]['errors'].append(error_value)

    algos = [a for a, _ in sorted(algo_stats_global.items(
    ), key=lambda x: np.mean(x[1]['average_aucs']), reverse=True)]
    print('algos', algos)
    algos = [a for a in algos if remap_name is None or a in remap_name]
    datasets = list(dataset_seed_aucs.keys())

    if custom_algo_order is None:
        custom_algo_order = sorted(algos)
    else:
        if remap_name is not None:
            inverse_remap = {v: k for k, v in remap_name.items()}
            print('inverse map', inverse_remap)
            custom_algo_order = [inverse_remap[a] for a in custom_algo_order]
    for i, algo in enumerate(custom_algo_order):
        if remap_name and algo not in remap_name:
            continue
        positions = np.array(range(len(datasets))) + i * bar_width
        average_aucs = algo_stats_global[algo]['average_aucs']
        errors = algo_stats_global[algo]['errors']
        name_algo = remap_name[algo] if remap_name and algo in remap_name else algo
        ax.bar(positions, average_aucs, bar_width, yerr=errors, label=name_algo, color=colormap.get(name_algo, None), capsize=5,
               alpha=0.8)
    # Final adjustments
    ax.set_xticks(np.arange(len(datasets)) + bar_width *
                  (len(algo_stats_global) - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=14)
    ax.legend(frameon=True, loc='lower right', bbox_to_anchor=(1.1, 0.0))
    ax.set_ylabel('Average AUC', fontsize=14)
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_title(
        plot_prefix_name + r'$\mathsf{'+mode+'}$ AUC', fontsize=16, weight='bold')
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    if max_y is None:
        max_y = 95 if mode == 'current' else 100
    ax.set_ylim([min_y, max_y])

    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')


def plot_learning_curve_bars(seed_aucs, title_name=None, ax=None, remap_name=None):
    if title_name is None:
        title_name = 'Aggregated AUC'

    if ax is None:
        fig, ax = plt.subplots()

    # Sort the algo_stats dictionary by average AUC in descending order
    # sorted_algo_stats = sorted(algo_stats.items(), key=lambda x: x[1]['average_auc'], reverse=True)

    # palette = plt.get_cmap('tab10').colors
    # print('algo:', algo)
    # algo_colors = {algo: palette[i % len(palette)] for i, algo in enumerate(sorted(algo_stats, key=lambda x: x[1]['average_auc'], reverse=True))}

    algo_stats = get_auc_stats(seed_aucs)

    # Sort the algo_stats dictionary by average AUC in descending order and prepare for plotting
    sorted_algo_stats = sorted(
        algo_stats.items(), key=lambda x: x[1]['average_auc'], reverse=True)

    # Create a color mapping for each algorithm based on the sorted order
    palette = plt.get_cmap('tab10').colors
    palette = sns.color_palette("husl", len(sorted_algo_stats))

    algo_colors = {algo: palette[i % len(palette)]
                   for i, (algo, _) in enumerate(sorted_algo_stats)}

    # Plot the bars with standard deviation as error bars
    for i, (algo, stats) in enumerate(sorted_algo_stats):
        if remap_name is not None and algo not in remap_name:
            continue

        name_algo = algo if remap_name is None else remap_name[algo]
        ax.bar(name_algo, stats['average_auc'],
               yerr=stats['std_auc'], color=algo_colors[algo], capsize=5)

    ax.tick_params(axis='x', rotation=45)  # Rotate labels
    # Align labels to the right
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")

    ax.set_title(title_name)
    ax.set_ylim([60, 95])
    ax.grid(True, which='major', linestyle='--', alpha=0.5)


def plot_learning_curve_dataset(dataset_agg_dfs, remap_name=None, colormap=None,
                                mode='avg', save_fig_path=None, error_type='std',
                                metric='test_acc'):
    fig, ax = plt.subplots(1, len(dataset_agg_dfs.keys()), figsize=(30, 10))
    handles, labels = [], []

    # Ensure ax is always iterable
    if not isinstance(ax, np.ndarray):
        ax = [ax]  # Make it a list so it's always subscriptable

    for i, (dataset, agg_df) in enumerate(dataset_agg_dfs.items()):
        plot_agg_over_seeds(agg_df, title_name=dataset,
                            ax=ax[i], metric=metric, remap_name=remap_name, colormap=colormap,
                            error_type=error_type)
        # Collect handles and labels for the current axis
        for handle, label in zip(*ax[i].get_legend_handles_labels()):
            if label not in labels:  # Check to avoid duplicates
                handles.append(handle)
                labels.append(label)

    # After plotting is done, you set common labels and a super title like so:
    # fig.suptitle(
    #     r'Test $\mathsf{'+mode+'}$ Accuracy Learning Curves', fontsize=30, weight='bold')

    fig.text(0.5, 0.02, 'Epoch', ha='center',
             va='center', fontsize=30, weight='bold')
    fig.text(0.02, 0.5, 'Test Accuracy', ha='center', va='center',
             rotation='vertical', fontsize=30, weight='bold')
    # fig.legend(handles, labels, loc='lower right', fontsize=30,
    #            frameon=True, bbox_to_anchor=(1.1, 0.0))
    fig.legend(handles, labels, loc='lower center', fontsize=30,
           frameon=True, ncol=4, bbox_to_anchor=(0.5, -0.16), markerscale=3)    # Adjust the rect to make space for the common title and labels
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')


def make_table(pivot_m):
    table = PrettyTable()
    columns = ['Base', 'Algorithm'] + \
        [col for col in pivot_m.columns if col not in ('base', 'algorithm')]
    table.field_names = columns

    max_values = pivot_m.max()

    for _, row in pivot_m.iterrows():
        row_data = []
        for col in pivot_m.columns:
            if col not in ('base', 'algorithm') and not pd.isna(row[col]):
                # Bold the highest value for each dataset
                if row[col] == max_values[col]:
                    row_data.append(f"**{row[col]:.5f}**")
                else:
                    row_data.append(f"{row[col]:.5f}")
            else:
                row_data.append(row[col])
        table.add_row(row_data)
    return table


def make_table_v2(df, remap_name=None, error_type='std'):
    if error_type not in ['std', 'sem']:
        raise ValueError("error_type must be either 'std' or 'sem'")

    # Pivot the dataframe for mean values
    pivot_mean_df = df.pivot(index='algo', columns='dataset', values='mean')

    # Pivot the dataframe for error values
    pivot_error_df = df.pivot(
        index='algo', columns='dataset', values=error_type)

    # Start building the HTML table
    html = '<table><tr><th>Algorithm</th>'
    for dataset in pivot_mean_df.columns:
        html += f'<th>{dataset}</th>'
    html += '</tr>'

    # Find the maximum mean values for each dataset
    max_values = pivot_mean_df.max()

    for index, row in pivot_mean_df.iterrows():
        if remap_name is not None and index not in remap_name:
            continue
        html += f'<tr><td>{index if remap_name is None else remap_name[index]}</td>'
        for dataset in pivot_mean_df.columns:
            mean_value = row[dataset]
            error_value = pivot_error_df.loc[index, dataset]
            # Bold the best value using HTML <b> tag
            if mean_value == max_values[dataset]:
                html += f'<td><b>{mean_value:.5f} +/- {error_value:.2f}</b></td>'
            else:
                html += f'<td>{mean_value:.5f} +/- {error_value:.2f}</td>'
        html += '</tr>'
    html += '</table>'

    # Display the HTML table in Jupyter Notebook
    display(HTML(html))

    return pivot_mean_df
