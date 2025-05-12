import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib import rcParams

# benchmark_win_rates = pd.read_csv('benchmark_win_rates.csv')
model_win_rates_pareto = pd.read_csv('model_win_rates_pareto.csv')
model_win_rates_max = pd.read_csv('model_win_rates_max.csv')
# grouped_df = benchmark_win_rates.groupby('benchmark_name')
# dfs_dict = {category: group for category, group in grouped_df}
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
    # Sort by win rate
    sorted_df = model_win_rates.sort_values('overall_win_rate', ascending=False)
    # Create bar plot
    ax = sns.barplot(x='model_name_short', y='overall_win_rate', data=sorted_df, hue='model_name_short', palette='magma')
    
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

# def identify_pareto_optimal(df, x_col, y_col, minimize_x=True, maximize_y=True):
#     """
#     Identify Pareto optimal points (best y for a given x).
    
#     Args:
#         df: DataFrame with metrics
#         x_col: Column name for x metric (e.g., latency, cost)
#         y_col: Column name for y metric (e.g., win_rate, accuracy)
#         minimize_x: Whether to minimize the x metric (True for cost/latency)
#         maximize_y: Whether to maximize the y metric (True for win_rate/accuracy)
        
#     Returns:
#         DataFrame with Pareto optimal flag
#     """
#     df = df.copy()
    
#     # Sort by x (ascending if minimize_x, descending otherwise) 
#     # and y (descending if maximize_y, ascending otherwise)
#     x_ascending = minimize_x
#     y_ascending = not maximize_y
    
#     df = df.sort_values([x_col, y_col], ascending=[x_ascending, y_ascending])
    
#     # Initialize Pareto frontier with the first point
#     pareto_frontier = [df.iloc[0]]
    
#     # Iterate through remaining points
#     for i in range(1, len(df)):
#         current_point = df.iloc[i]
#         last_pareto = pareto_frontier[-1]
        
#         # If current point has better y than the last Pareto optimal point,
#         # add it to the Pareto frontier
#         if (maximize_y and current_point[y_col] > last_pareto[y_col]) or \
#            (not maximize_y and current_point[y_col] < last_pareto[y_col]):
#             pareto_frontier.append(current_point)
    
#     # Create a DataFrame from the Pareto frontier
#     pareto_df = pd.DataFrame(pareto_frontier)
    
#     # Add a flag to the original DataFrame indicating Pareto optimal points
#     df['pareto_optimal'] = df.index.isin(pareto_df.index)
    
#     return df

# def plot_pareto_frontier(df, x_col, y_col, title, x_label, y_label, filename, 
#                          minimize_x=True, maximize_y=True, model_col='model_name_short'):
#     """
#     Plot the Pareto frontier.
    
#     Args:
#         df: DataFrame with metrics
#         x_col: Column name for x metric (e.g., latency, cost)
#         y_col: Column name for y metric (e.g., win_rate, accuracy)
#         title: Plot title
#         x_label: x-axis label
#         y_label: y-axis label
#         filename: Output file name
#         minimize_x: Whether to minimize the x metric (True for cost/latency)
#         maximize_y: Whether to maximize the y metric (True for win_rate/accuracy)
#         model_col: Column name for model/agent names
#     """

#     # Set a more attractive style
#     plt.style.use('seaborn-v0_8-whitegrid')
#     mpl.rcParams['font.family'] = 'sans-serif'
#     mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
#     mpl.rcParams['axes.labelsize'] = 12
#     mpl.rcParams['axes.titlesize'] = 14
#     mpl.rcParams['xtick.labelsize'] = 10
#     mpl.rcParams['ytick.labelsize'] = 10

#     # Identify Pareto optimal points
#     pareto_df = identify_pareto_optimal(df, x_col, y_col, minimize_x, maximize_y)
    
#     # Create a custom colormap for the points
#     colors = ["#3498db", "#e74c3c"]  # Blue for non-optimal, Red for optimal
#     cmap = LinearSegmentedColormap.from_list("pareto_cmap", colors, N=2)
    
#     fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
#     # Add a subtle background gradient
#     ax.set_facecolor('#f8f9fa')
    
#     # Plot non-optimal points first
#     non_optimal = pareto_df[~pareto_df['pareto_optimal']]
#     optimal = pareto_df[pareto_df['pareto_optimal']]
    
#     # Plot non-optimal points
#     ax.scatter(
#         non_optimal[x_col], 
#         non_optimal[y_col], 
#         s=120,
#         color='#3498db',
#         alpha=0.7,
#         edgecolor='white',
#         linewidth=1,
#         label='Non-Optimal'
#     )
    
#     # Plot optimal points
#     ax.scatter(
#         optimal[x_col], 
#         optimal[y_col], 
#         s=150,
#         color='#e74c3c',
#         alpha=0.9,
#         edgecolor='white',
#         linewidth=1.5,
#         marker='D',
#         label='Pareto Optimal'
#     )
    
#     # Connect Pareto optimal points with a line
#     optimal_points = optimal.sort_values(x_col)
#     ax.plot(
#         optimal_points[x_col], 
#         optimal_points[y_col], 
#         color='#e74c3c', 
#         linestyle='--',
#         linewidth=2.5,
#         alpha=0.8
#     )
    
#     # Add model/agent names as labels with better styling
#     for _, row in pareto_df.iterrows():
#         color = '#e74c3c' if row['pareto_optimal'] else '#3498db'
#         weight = 'bold' if row['pareto_optimal'] else 'normal'
#         ax.annotate(
#             row[model_col],
#             (row[x_col], row[y_col]),
#             xytext=(7, 7),
#             textcoords='offset points',
#             fontsize=10,
#             color=color,
#             weight=weight,
#             bbox=dict(
#                 boxstyle="round,pad=0.3",
#                 fc='white',
#                 ec=color,
#                 alpha=0.7
#             )
#         )
    
#     # Add title and labels with better styling
#     ax.set_title(title, fontsize=16, pad=20, weight='bold')
#     ax.set_xlabel(x_label, fontsize=14, labelpad=10)
#     ax.set_ylabel(y_label, fontsize=14, labelpad=10)
    
#     # Improve grid appearance
#     ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
#     # Add a legend with better styling
#     legend = ax.legend(
#         loc='best',
#         frameon=True,
#         framealpha=0.95,
#         facecolor='white',
#         edgecolor='lightgray',
#         fontsize=12
#     )
    
#     # Add a subtle border around the plot
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_color('lightgray')
#         spine.set_linewidth(0.5)
    
#     plt.tight_layout()
    
#     # Save with higher quality
#     plt.savefig(f'visualizations/{filename}', dpi=300, bbox_inches='tight')
#     print(f"Saved file: visualizations/{filename}")
    
#     return pareto_df

# def grid_pareto_frontier_by_benchmark(tasks, merged_df, x_col, y_col, x_label, y_label, 
#                                      num_cols, filename, minimize_x=True, maximize_y=True, 
#                                      model_col='model_name_short'):
#     """
#     Create Pareto frontier plots in a grid for each benchmark
    
#     Args:
#         tasks: names of benchmarks to group by
#         merged_df: dataframe with data to plot
#         x_col: Column name for x metric (e.g., latency, cost)
#         y_col: Column name for y metric (e.g., win_rate, accuracy)
#         x_label: x-axis label
#         y_label: y-axis label
#         num_cols: number of columns in grid
#         filename: final plot name
#         minimize_x: Whether to minimize the x metric (True for cost/latency)
#         maximize_y: Whether to maximize the y metric (True for win_rate/accuracy)
#         model_col: Column name for model/agent names
#     """
#     # Calculate the number of tasks
#     n_tasks = len(tasks)
    
#     # If only one task or no tasks, create a single plot
#     if n_tasks <= 1:
#         if n_tasks == 1:
#             task = tasks[0]
#             df_t = merged_df[merged_df['benchmark_name'] == task]
#             title = f"Pareto Frontier: {task} - {y_label} vs. {x_label}"
#         else:
#             # If no tasks, use all data
#             df_t = merged_df
#             title = f"Pareto Frontier: All Benchmarks - {y_label} vs. {x_label}"
        
#         # Use the existing plot_pareto_frontier function for a single plot
#         pareto_df = plot_pareto_frontier(
#             df_t, x_col, y_col, title, x_label, y_label, 
#             filename, minimize_x, maximize_y, model_col
#         )
        
#         return pareto_df
    
#     # Multiple tasks - create a grid of subplots
#     n_cols = min(num_cols, n_tasks)  # Maximum number of columns
#     n_rows = (n_tasks + n_cols - 1) // n_cols  
    
#     # Create a single figure with subplots
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), facecolor='white')
    
#     # Add a title to the entire figure
#     fig.suptitle(f"Pareto Frontiers: {y_label} vs. {x_label} by Benchmark", 
#                  fontsize=18, weight='bold', y=0.98)
    
#     # Flatten the axes array for easy iteration
#     axes = axes.flatten()
    
#     all_pareto_dfs = []
    
#     for idx, task in enumerate(tasks):
#         ax = axes[idx]
#         df_t = merged_df[merged_df['benchmark_name'] == task]
#         benchmark_name = task
        
#         # Skip if no data for this benchmark
#         if len(df_t) == 0:
#             ax.set_visible(False)
#             continue
        
#         # Set background color
#         ax.set_facecolor('#f8f9fa')
        
#         # Identify Pareto optimal points for this benchmark
#         pareto_df = identify_pareto_optimal(df_t, x_col, y_col, minimize_x, maximize_y)
#         all_pareto_dfs.append(pareto_df)
        
#         # Plot non-optimal points first
#         non_optimal = pareto_df[~pareto_df['pareto_optimal']]
#         optimal = pareto_df[pareto_df['pareto_optimal']]
        
#         # Plot non-optimal points
#         ax.scatter(
#             non_optimal[x_col], 
#             non_optimal[y_col], 
#             s=100,
#             color='#3498db',
#             alpha=0.7,
#             edgecolor='white',
#             linewidth=1,
#             label='Non-Optimal'
#         )
        
#         # Plot optimal points
#         ax.scatter(
#             optimal[x_col], 
#             optimal[y_col], 
#             s=120,
#             color='#e74c3c',
#             alpha=0.9,
#             edgecolor='white',
#             linewidth=1.5,
#             marker='D',
#             label='Pareto Optimal'
#         )
        
#         # Connect Pareto optimal points with a line
#         optimal_points = optimal.sort_values(x_col)
#         ax.plot(
#             optimal_points[x_col], 
#             optimal_points[y_col], 
#             color='#e74c3c', 
#             linestyle='--',
#             linewidth=2,
#             alpha=0.8
#         )
        
#         # Add model/agent names as labels with better styling
#         for _, row in pareto_df.iterrows():
#             color = '#e74c3c' if row['pareto_optimal'] else '#3498db'
#             weight = 'bold' if row['pareto_optimal'] else 'normal'
#             fontsize = 9 if row['pareto_optimal'] else 8
            
#             # Only add box around Pareto optimal points to reduce clutter
#             if row['pareto_optimal']:
#                 bbox_props = dict(
#                     boxstyle="round,pad=0.3",
#                     fc='white',
#                     ec=color,
#                     alpha=0.7
#                 )
#             else:
#                 bbox_props = None
                
#             ax.annotate(
#                 row[model_col],
#                 (row[x_col], row[y_col]),
#                 xytext=(7, 7),
#                 textcoords='offset points',
#                 fontsize=fontsize,
#                 color=color,
#                 weight=weight,
#                 bbox=bbox_props
#             )
        
#         # Add title and labels with better styling
#         ax.set_title(benchmark_name, fontsize=14, pad=10, weight='bold')
#         ax.set_xlabel(x_label, fontsize=12, labelpad=8)
#         ax.set_ylabel(y_label, fontsize=12, labelpad=8)
        
#         # Improve grid appearance
#         ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
#         # Add a subtle border around the plot
#         for spine in ax.spines.values():
#             spine.set_visible(True)
#             spine.set_color('lightgray')
#             spine.set_linewidth(0.5)
    
#     # Add a single legend for the entire figure
#     handles, labels = axes[0].get_legend_handles_labels()
#     if handles:
#         legend = fig.legend(
#             handles, 
#             labels,
#             loc='lower center',
#             bbox_to_anchor=(0.5, 0.01),
#             ncol=2,
#             frameon=True,
#             framealpha=0.95,
#             facecolor='white',
#             edgecolor='lightgray',
#             fontsize=12
#         )
    
#     # Hide any unused subplots
#     for idx in range(n_tasks, len(axes)):
#         axes[idx].set_visible(False)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for annotations and title
#     plt.savefig(f'visualizations/new_plots/{filename}', dpi=300, bbox_inches='tight')
#     print(f"Saved file: visualizations/new_plots/{filename}")
    
#     # Combine all pareto dataframes
#     if all_pareto_dfs:
#         combined_pareto_df = pd.concat(all_pareto_dfs, ignore_index=True)
#         return combined_pareto_df
#     else:
#         return None

# def grid_scatter_by_benchmark(tasks, merged_df, x_axis, y_axis, x_label, y_label, num_cols, filename):
#     """
#     Create scatter plots in a grid comparing metrics for each benchmark
    
#     Args:
#         tasks: names of benchmarks to group by
#         merged_df: dataframe with data to plot
#         x_axis: metric 1 being compared
#         y_axis: metric 2 being compared
#         x_label: x-axis label
#         y_label: y-axis label
#         num_cols: number of columns in grid
#         filename: final plot name
#     """

#     # Calculate the grid dimensions based on the number of tasks
#     n_tasks = len(tasks)
    
#     # If only one task, create a single plot
#     if n_tasks <= 1:
#         fig, ax = plt.subplots(figsize=(12, 8))
        
#         if n_tasks == 1:
#             task = tasks[0]
#             df_t = merged_df[merged_df['benchmark_name'] == task]
#             benchmark_name = task
            
#             # Create scatter plot
#             scatter = ax.scatter(df_t[x_axis], df_t[y_axis], alpha=0.5)
            
#             # Annotate each point with the model name
#             for i, row in df_t.iterrows():
#                 ax.annotate(row['model_name_short'], 
#                             (row[x_axis], row[y_axis]),
#                             xytext=(5, 5),
#                             textcoords='offset points',
#                             fontsize=10)
            
#             ax.set_title(benchmark_name)
#         else:
#             # If no tasks, use all data
#             scatter = ax.scatter(merged_df[x_axis], merged_df[y_axis], alpha=0.5)
            
#             # Annotate each point with the model name
#             for i, row in merged_df.iterrows():
#                 ax.annotate(row['model_name_short'], 
#                             (row[x_axis], row[y_axis]),
#                             xytext=(5, 5),
#                             textcoords='offset points',
#                             fontsize=10)
            
#             ax.set_title("All Benchmarks")
        
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
        
#     else:
#         # Multiple tasks - create a grid of subplots
#         n_cols = min(num_cols, n_tasks)  # Maximum number of columns
#         n_rows = (n_tasks + n_cols - 1) // n_cols  
        
#         # Create a single figure with subplots
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(35, 5 * n_rows))
        
#         # Flatten the axes array for easy iteration
#         axes = axes.flatten()

#         for idx, task in enumerate(tasks):
#             ax = axes[idx]
#             df_t = merged_df[merged_df['benchmark_name'] == task]
#             benchmark_name = task 
        
#             # Create scatter plot
#             scatter = ax.scatter(df_t[x_axis], df_t[y_axis], alpha=0.5)
        
#             # Annotate each point with the model name
#             for i, row in df_t.iterrows():
#                 ax.annotate(row['model_name_short'], 
#                             (row[x_axis], row[y_axis]),
#                             xytext=(5, 5),
#                             textcoords='offset points',
#                             fontsize=8)
        
#             ax.set_title(benchmark_name)
#             ax.set_xlabel(x_label)
#             ax.set_ylabel(y_label)
        
#         # Hide any unused subplots
#         for idx in range(n_tasks, len(axes)):
#             axes[idx].set_visible(False)
#     plt.tight_layout()  # Adjust layout to make room for annotations
#     plt.savefig(f'visualizations/{filename}', dpi=300)
#     print(f"Saved file: visualizations/{filename}")

def create_heatmaps(agent_scaffold, metric, title, x_label, y_label, legend_name):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    df = pd.read_csv('cleaned_all_metrics.csv')
    cols = ['benchmark_name', 'agent_name_short', 'model_name_short', 'total_cost']
    df = df[cols].copy()
    # Only keep rows with generalist agents
    if agent_scaffold == 'generalist':
        generalist_scaffold = 'HAL Generalist Agent'
        df = df[df['agent_name_short'] == generalist_scaffold]
        print(df.head())
    # only keep rows with task specific agents
    elif agent_scaffold == 'task_specific':
        task_specific_scaffolds = ['Col-bench Text', 'HF Open Deep Research', 'Scicode Tool Calling Agent', 'Scicode Zero Shot Agent', 'SAB Example Agent', 'SWE-Agent', 'TAU-bench FewShot', 'USACO Episodic + Semantic']
        df = df[df['agent_name_short'].isin(task_specific_scaffolds)]
        print(df.head())
    else:
        print("Error: scaffold type not found")
        return
    
    duplicates = df[df.duplicated(subset=['benchmark_name', 'model_name_short'], keep=False)]
    print(duplicates)
    
    df_pivot = df.pivot_table(columns='model_name_short', index='benchmark_name', values=metric, aggfunc='mean')
    
    # Calculate averages
    df_pivot['Average'] = df_pivot.mean(axis=1)  # Average for each benchmark (row)
    col_avg = df_pivot.drop(columns=['Average']).mean(axis=0)
    # Calculate the average of averages (grand mean)
    grand_mean = df_pivot.drop(columns=['Average']).values.mean()
    col_avg['Average'] = grand_mean  # Set the bottom-right cell to the grand mean
    df_pivot.loc['Average'] = col_avg  # Average for each model (column)
    
    # Create a custom colormap
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    
    # Create the figure with appropriate size (only create one figure)
    plt.figure(figsize=(11, 4))
    
    # Create the main heatmap (without averages)
    main_data = df_pivot.iloc[:-1, :-1]
    ax = sns.heatmap(
        data=main_data,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': legend_name}
    )
    
    # 2. Add the row averages (right column)
    for i in range(len(df_pivot.index) - 1):
        # Add white cell with grid
        rect = plt.Rectangle(
            (main_data.shape[1], i), 
            1, 1, 
            fill=True, 
            facecolor='white',
            edgecolor='lightgray', 
            lw=0.5
        )
        ax.add_patch(rect)
        # Add text
        ax.text(
            main_data.shape[1] + 0.5, 
            i + 0.5, 
            f'{df_pivot.iloc[i, -1]:.2f}', 
            ha='center', 
            va='center',
            fontweight='bold'
        )
    
    # 3. Add the column averages (bottom row)
    for j in range(len(df_pivot.columns) - 1):
        # Add white cell with grid
        rect = plt.Rectangle(
            (j, main_data.shape[0]), 
            1, 1, 
            fill=True, 
            facecolor='white',
            edgecolor='lightgray', 
            lw=0.5
        )
        ax.add_patch(rect)
        # Add text
        ax.text(
            j + 0.5, 
            main_data.shape[0] + 0.5, 
            f'{df_pivot.iloc[-1, j]:.2f}', 
            ha='center', 
            va='center',
            fontweight='bold'
        )
    
    # 4. Add the bottom-right corner (average of averages)
    # Add white cell with grid
    rect = plt.Rectangle(
        (main_data.shape[1], main_data.shape[0]), 
        1, 1, 
        fill=True, 
        facecolor='white',
        edgecolor='lightgray', 
        lw=0.5
    )
    ax.add_patch(rect)
    # Add text
    ax.text(
        main_data.shape[1] + 0.5, 
        main_data.shape[0] + 0.5, 
        f'{df_pivot.iloc[-1, -1]:.2f}', 
        ha='center', 
        va='center',
        fontweight='bold'
    )
    
    # 5. Add "Average" label to the right column and bottom row
    ax.text(
        main_data.shape[1] + 0.5, 
        -0.3, 
        'Average', 
        ha='center', 
        va='center',
        fontweight='bold',
        fontsize=10
    )
    ax.text(
        -0.3, 
        main_data.shape[0] + 0.5, 
        'Average', 
        ha='center', 
        va='center',
        fontweight='bold',
        fontsize=10
    )
    

    # Customize the appearance
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'visualizations/new_plots/{agent_scaffold}_{metric}_heatmap.png', bbox_inches='tight', dpi=300)



# # def model_cost_win_rate():
# #     model_costs = pd.read_csv('model_total_usage.csv')
# #     model_winrates = pd.read_csv('benchmark_win_rates.csv')
# #     df_m = model_winrates.merge(model_costs, on=['model_name_short', 'benchmark_name'], how='left')
# #     tasks = df_m['benchmark_name'].unique()
# #     grid_pareto_frontier_by_benchmark(tasks, df_m, 'total_cost', 'win_rate_mean', 'Total Cost', 'Mean Win Rate', 4, 'cost_win_rate.png')

# def cost_accuracy():
#     model_costs = pd.read_csv('model_total_usage.csv')
#     model_accuracy = pd.read_csv('model_accuracy.csv')
#     df_m = model_accuracy.merge(model_costs, on=['model_name_short', 'benchmark_name'], how='left')
#     tasks = df_m['benchmark_name'].unique()
#     grid_pareto_frontier_by_benchmark(tasks, df_m, 'total_cost', 'accuracy', 'Total Cost', 'Accuracy', 4, 'model_cost_accuracy.png')

# def latency_accuracy():
#     model_latency = pd.read_csv('model_latency.csv')
#     model_accuracy = pd.read_csv('model_accuracy.csv')
#     df_m = model_accuracy.merge(model_latency, on=['model_name_short', 'benchmark_name'], how='left')
#     tasks = df_m['benchmark_name'].unique()
#     grid_pareto_frontier_by_benchmark(tasks, df_m, 'latency', 'accuracy', 'Mean Latency', 'Accuracy', 4, 'model_latency_accuracy.png')

# def cost_win_rate():
#     # with model win rates and data/model_mean_cost, plot using plot_pareto_fronteir function
#     model_mean_costs = pd.read_csv('data/model_mean_cost.csv')
#     df_m = pd.merge(model_mean_costs, model_win_rates, on='model_name_short', how='inner')
#     cols = ['model_name_short', 'mean_cost', 'win_rate_mean', 'overall_win_rate']
#     df_m = df_m[cols].copy()
#     plot_pareto_frontier(df_m, 'mean_cost', 'overall_win_rate', 'Pareto Frontier: Win Rate vs. Mean Cost', 'Mean Cost', 'Win Rate', 'new_plots/cost_win_rate.png')

# def latency_win_rate():
#     model_mean_latency = pd.read_csv('data/model_mean_latency.csv')
#     df_m = pd.merge(model_mean_latency, model_win_rates, on='model_name_short', how='inner')
#     cols = ['model_name_short', 'mean_latency', 'win_rate_mean', 'overall_win_rate']
#     df_m = df_m[cols].copy()
#     plot_pareto_frontier(df_m, 'mean_latency', 'overall_win_rate', 'Pareto Frontier: Win Rate vs. Mean Latency', 'Mean Latency', 'Win Rate', 'new_plots/latency_win_rate.png')

# model_win_rate_bar(model_win_rates_max, 'max')
# model_win_rate_bar(model_win_rates_pareto, 'pareto')
# model_accuracy_full()
# scaffold_accuracy()

create_heatmaps('generalist', 'total_cost', 'Total Costs of Generalist Agents', 'Model Name', 'Benchmark Name', 'Total Cost')

