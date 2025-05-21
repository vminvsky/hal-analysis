import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
import matplotlib as mpl
from matplotlib import rcParams
import ast

model_win_rates_pareto = pd.read_csv('model_win_rates_pareto.csv')
model_win_rates_max = pd.read_csv('model_win_rates_max.csv')
sns.set_theme(style="whitegrid", palette="muted") 

def model_accuracy_full():
    model_accuracy = pd.read_csv('model_accuracy.csv')
    grouped_df = model_accuracy.groupby('benchmark_name')
    dfs_dict = {category: group for category, group in grouped_df}
    sorted_groups = [group.sort_values(by='accuracy', ascending=False) for _, group in grouped_df]
    num_groups = len(sorted_groups)
    cols = 5  
    rows = (num_groups // cols) + (num_groups % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()
    for i, (group) in enumerate(sorted_groups):
        category = group['benchmark_name'].iloc[0]
        ax = axes[i]
        ax = sns.barplot(x='model_name_short', y='accuracy', data=group, hue='model_name_short', palette='magma', ax=ax)
        ax.set_title(f'{category}', fontsize=14)
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        for p in ax.patches:

            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=5, color='black', xytext=(0, 5), textcoords='offset points')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig('visualizations/new_plots/model_accuracy.png', dpi=300)

def scaffold_accuracy():
    scaffold_accuracy = pd.read_csv('benchmark_accuracy.csv')
    grouped_df = scaffold_accuracy.groupby('benchmark_name')
    dfs_dict = {category: group for category, group in grouped_df}
    sorted_groups = [group.sort_values(by='accuracy', ascending=False) for _, group in grouped_df]
    num_groups = len(sorted_groups)
    cols = 5  
    rows = (num_groups // cols) + (num_groups % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()
    for i, (group) in enumerate(sorted_groups):
        category = group['benchmark_name'].iloc[0]
        ax = axes[i]
        ax = sns.barplot(x='agent_name_short', y='accuracy', data=group, hue='agent_name_short', palette='magma', ax=ax)
        ax.set_title(f'{category}', fontsize=14)
        ax.set_xlabel('Agent Scaffold')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=5, color='black', xytext=(0, 5), textcoords='offset points')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig('visualizations/new_plots/scaffold_accuracy.png', dpi=300)

def model_win_rate_bar(model_win_rates, calc_type):
    """
    Plot the win rates for each model.
    
    Args:
        model_win_rates: DataFrame with win rates for each model
    """
    plt.figure(figsize=(12, 6))

    # Remove three models from win rate analysis
    models_to_remove = ['2.5-pro', 'o1', 'o3-mini']
    pattern = '|'.join(models_to_remove)
    model_win_rates_cleaned = model_win_rates[~model_win_rates['model_name_short'].str.contains(pattern, case=False, na=False)]

    # Sort by win rate
    sorted_df = model_win_rates_cleaned.sort_values('overall_win_rate', ascending=False)

    # Create bar plot
    ax = sns.barplot(x='model_name_short', y='overall_win_rate', data=sorted_df, hue='model_name_short', palette='dark:gray')
    norm = plt.Normalize(sorted_df['overall_win_rate'].min(), sorted_df['overall_win_rate'].max())
    cmap = plt.cm.magma_r
    colors = cmap(norm(sorted_df['overall_win_rate']))

    for bar, color in zip(ax.patches, colors):
        bar.set_facecolor(color)
    
    if calc_type == 'max':
        plt.title(f'Model Win Rates Calculated Using Max Accuracy')
    elif calc_type == 'pareto':
        plt.title(f'Model Win Rates Calculated Using Distance from Convex Hull')
    plt.xlabel('Model')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Add win rate values to bar
    for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

    # Add horizontal gridlines
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f'visualizations/new_plots/model_win_rates_{calc_type}.png')

def create_heatmaps(df, agent_scaffold, metric, title, x_label, y_label, legend_name):
    # Only keep rows with generalist agents
    if agent_scaffold == 'generalist':
        cols = ['benchmark_name', 'agent_name_short', 'model_name_short', metric]
        df = df[cols].copy()
        generalist_scaffold = 'HAL Generalist Agent'
        df = df[df['agent_name_short'] == generalist_scaffold]
    # only keep rows with task specific agents
    elif agent_scaffold == 'task_specific':
        cols = ['benchmark_name', 'agent_name_short', 'model_name_short', metric]
        df = df[cols].copy()
        task_specific_scaffolds = ['Col-bench Text', 'HF Open Deep Research', 'Scicode Tool Calling Agent', 'Scicode Zero Shot Agent', 'SAB Example Agent', 'SWE-Agent', 'TAU-bench FewShot', 'USACO Episodic + Semantic', 'CORE-Agent', 'Assistantbench Browser Agent']
        df = df[df['agent_name_short'].isin(task_specific_scaffolds)]
    elif agent_scaffold == 'max_acc' or agent_scaffold == 'dist':
        cols = ['benchmark_name', 'model_name_short', metric]
        df = df[cols].copy()
        print("Creating win rate heatmaps")
    else:
        print("Error: scaffold type not found")
        return
    
    # duplicates = df[df.duplicated(subset=['benchmark_name', 'model_name_short'], keep=False)]
    # print(duplicates)
    
    df_pivot = df.pivot_table(columns='model_name_short', index='benchmark_name', values=metric, aggfunc='mean')

    # normalize by row for color gradient
    # norm = df_pivot.sub(df_pivot.min(axis=1), axis=0)
    # norm = norm.div(df_pivot.max(axis=1) - df_pivot.min(axis=1), axis=0)
    
    # Create a custom colormap
    base_cmap = plt.colormaps.get_cmap('Purples')
    colors = base_cmap(np.linspace(0,1,256))
    colors = colors[50:]
    custom_purples =LinearSegmentedColormap.from_list("custom_purples", colors)
    custom_purples.set_bad(color='#404040')

    # Create greyed out boxes for NaN values
    # annot = df_pivot.copy()
    # annot = annot.map(lambda x: "no runs" if pd.isna(x) else f"{x:.2f}")
    
    fig, ax = plt.subplots(figsize=(11, 5))

    # Create the main heatmap 
    sns.heatmap(
        data=df_pivot,
        annot=df_pivot,
        annot_kws={"fontsize":7},
        fmt='.2f',
        cmap=custom_purples,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': legend_name, 'orientation': 'vertical'},
    )
    
    plt.tight_layout()

    for i in range(df_pivot.shape[0]):
        for j in range(df_pivot.shape[1]):
            if pd.isna(df_pivot.iloc[i,j]):
                ax.text(j+0.5, i+0.5, 'no runs', ha='center', va='center', color='white', fontsize=7)
    
    # Customize the appearance
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Adjust layout
    plt.savefig(f'visualizations/new_plots/heatmaps/{agent_scaffold}_{metric}_heatmap.png', bbox_inches='tight', dpi=300)


model_win_rate_bar(model_win_rates_max, 'max')
model_win_rate_bar(model_win_rates_pareto, 'pareto')

cleaned_dataset = pd.read_csv('cleaned_all_metrics.csv')
win_rates_max = pd.read_csv("benchmark_win_rates_max.csv")
win_rates_pareto = pd.read_csv("benchmark_win_rates_pareto.csv")
win_rates = pd.read_csv("benchmark_win_rates.csv")

create_heatmaps(cleaned_dataset, 'generalist', 'total_cost', 'Total Costs of Generalist Agents', 'Model Name', 'Benchmark Name', 'Total Cost')
create_heatmaps(cleaned_dataset, 'generalist', 'accuracy', 'Accuracy of Generalist Agents', 'Model Name', 'Benchmark Name', 'Accuracy')
create_heatmaps(cleaned_dataset, 'task_specific', 'accuracy', 'Accuracy of Task Specific Agents', 'Model Name', 'Benchmark Name', 'Accuracy')
create_heatmaps(cleaned_dataset, 'task_specific', 'total_cost', 'Total Costs of Task Specific Agents', 'Model Name', 'Benchmark Name', 'Total Cost')
create_heatmaps(cleaned_dataset, 'generalist', 'mean_latency', 'Mean Latencies of Generalist Agents', 'Model Name', 'Benchmark Name', 'Mean Latency')
create_heatmaps(cleaned_dataset, 'task_specific', 'mean_latency', 'Mean Latencies of Task Specific Agents', 'Model Name', 'Benchmark Name', 'Mean Latency')
create_heatmaps(win_rates_max, 'max_acc', 'overall_win_rate', 'Win Rates Using Max Accuracy Across Agents', 'Model Name', 'Benchmark Name', 'Win Rate')
create_heatmaps(win_rates_pareto, 'dist', 'overall_win_rate', 'Win Rates Using Distance from Cost vs. Max Accuracy Pareto', 'Model Name', 'Benchmark Name', 'Win Rate')
create_heatmaps(win_rates, 'generalist', 'overall_win_rate', 'Win Rates of Generalist Agents', 'Model Name', 'Benchmark Name', 'Win Rate')
create_heatmaps(win_rates, 'task_specific', 'overall_win_rate', 'Win Rates of Task Specific Agents', 'Model Name', 'Benchmark Name', 'Win Rate')