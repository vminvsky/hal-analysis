import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.spatial import ConvexHull
import os
import csv

def cross(o, a, b):
    """
    Cross product for determining the convex hull.
    
    Args:
        o, a, b: Points for cross product calculation
        
    Returns:
        Cross product value
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def compute_hull_side(points):
    """
    Compute one side of the convex hull using the cross product test.
    
    Args:
        points: List of points as (x, y, index) tuples
        
    Returns:
        List of indices of points on the hull
    """
    hull = []
    for i, p in enumerate(points):
        while len(hull) >= 2 and cross(points[hull[-2]], points[hull[-1]], p) <= 0:
            hull.pop()
        hull.append(i)
    return [points[i][2] for i in hull]

def is_pareto_efficient(points, candidate_idx, x_col, y_col, minimize_x=True, maximize_y=True):
    """
    Determine if a point is Pareto efficient.
    
    Args:
        points: DataFrame with all points
        candidate_idx: Index of the candidate point
        x_col, y_col: Column names for metrics
        minimize_x, maximize_y: Optimization directions
        
    Returns:
        Boolean indicating if the point is Pareto efficient
    """
    candidate = points.iloc[candidate_idx]
    
    for i in range(len(points)):
        if i == candidate_idx:  # Skip comparing with itself
            continue
            
        other = points.iloc[i]
        
        # Check if other dominates candidate
        x_better = (minimize_x and other[x_col] <= candidate[x_col]) or \
                   (not minimize_x and other[x_col] >= candidate[x_col])
                   
        y_better = (maximize_y and other[y_col] >= candidate[y_col]) or \
                   (not maximize_y and other[y_col] <= candidate[y_col])
                   
        strictly_better = (minimize_x and other[x_col] < candidate[x_col]) or \
                          (not minimize_x and other[x_col] > candidate[x_col]) or \
                          (maximize_y and other[y_col] > candidate[y_col]) or \
                          (not maximize_y and other[y_col] < candidate[y_col])
                          
        if x_better and y_better and strictly_better:
            return False
            
    return True

def identify_pareto_optimal(df, x_col, y_col, minimize_x=True, maximize_y=True):
    """
    Identify Pareto optimal points (best y for a given x).
    
    Args:
        df: DataFrame with metrics
        x_col: Column name for x metric (e.g., latency, cost)
        y_col: Column name for y metric (e.g., win_rate, accuracy)
        minimize_x: Whether to minimize the x metric (True for cost/latency)
        maximize_y: Whether to maximize the y metric (True for win_rate/accuracy)
        
    Returns:
        DataFrame with Pareto optimal flag
    """
    df = df.copy()
    
    # Initialize all points as non-optimal
    df['pareto_optimal'] = False
    
    # If we have fewer than 2 points, they're all optimal
    if len(df) <= 1:
        df['pareto_optimal'] = True
        return df
    
    # Extract points as (x, y, index) tuples
    # Transform values based on minimize/maximize preferences
    points = []
    for i in range(len(df)):
        x_val = df.iloc[i][x_col] if minimize_x else -df.iloc[i][x_col]
        y_val = df.iloc[i][y_col] if maximize_y else -df.iloc[i][y_col]
        points.append((x_val, y_val, i))
    
    # Sort points by x (ascending)
    points.sort()
    
    # Compute the upper convex hull (which becomes the Pareto frontier)
    # Reverse the points as in test_analysis.py
    hull_indices = compute_hull_side(list(reversed(points)))
    
    # Filter points on the hull to ensure they are truly Pareto efficient
    pareto_indices = []
    for idx in hull_indices:
        if is_pareto_efficient(df, idx, x_col, y_col, minimize_x, maximize_y):
            pareto_indices.append(idx)
    
    # Mark points on the Pareto frontier in the original DataFrame
    for idx in pareto_indices:
        df.iloc[idx, df.columns.get_loc('pareto_optimal')] = True
    
    print(df.head())
    
    return df

def plot_pareto_frontier(df, x_col, y_col, title, x_label, y_label, filename, 
                         minimize_x=True, maximize_y=True, model_col='model_name_short'):
    """
    Plot the Pareto frontier.
    
    Args:
        df: DataFrame with metrics
        x_col: Column name for x metric (e.g., latency, cost)
        y_col: Column name for y metric (e.g., win_rate, accuracy)
        title: Plot title
        x_label: x-axis label
        y_label: y-axis label
        filename: Output file name
        minimize_x: Whether to minimize the x metric (True for cost/latency)
        maximize_y: Whether to maximize the y metric (True for win_rate/accuracy)
        model_col: Column name for model/agent names
    """
    # Ensure visualizations directory exists
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('visualizations/auc_data', exist_ok=True)

    # Set a more attractive style
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10

    # Identify Pareto optimal points
    pareto_df = identify_pareto_optimal(df, x_col, y_col, minimize_x, maximize_y)
    
    # Create a custom colormap for the points
    colors = ["#3498db", "#e74c3c"]  # Blue for non-optimal, Red for optimal
    cmap = LinearSegmentedColormap.from_list("pareto_cmap", colors, N=2)
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    # Add a subtle background gradient
    ax.set_facecolor('#f8f9fa')
    
    # Plot non-optimal points first
    non_optimal = pareto_df[~pareto_df['pareto_optimal']]
    optimal = pareto_df[pareto_df['pareto_optimal']]
    
    # Plot non-optimal points
    ax.scatter(
        non_optimal[x_col], 
        non_optimal[y_col], 
        s=120,
        color='#3498db',
        alpha=0.7,
        edgecolor='white',
        linewidth=1,
        label='Non-Optimal'
    )
    
    # Plot optimal points
    ax.scatter(
        optimal[x_col], 
        optimal[y_col], 
        s=150,
        color='#e74c3c',
        alpha=0.9,
        edgecolor='white',
        linewidth=1.5,
        marker='D',
        label='Pareto Optimal'
    )
    
    # Connect Pareto optimal points with a line
    optimal_points = optimal.sort_values(x_col)
    ax.plot(
        optimal_points[x_col], 
        optimal_points[y_col], 
        color='#e74c3c', 
        linestyle='--',
        linewidth=2.5,
        alpha=0.8
    )
    
    # Plot convex hull if there are enough points
    # if len(optimal_points) >= 3:
    #     # Get coordinates for convex hull
    #     points = np.column_stack([optimal_points[x_col].values, optimal_points[y_col].values])
    #     hull = ConvexHull(points)
        
    #     # Get hull vertices in order
    #     hull_vertices = []
    #     for vertex in hull.vertices:
    #         hull_vertices.append(points[vertex])
    #     hull_vertices.append(hull_vertices[0])  # Close the loop
    #     hull_vertices = np.array(hull_vertices)
        
        # # Plot the convex hull
        # ax.plot(
        #     hull_vertices[:, 0],
        #     hull_vertices[:, 1],
        #     color='#2ecc71',  # Green color for convex hull
        #     linestyle='-',
        #     linewidth=2,
        #     alpha=0.7,
        #     label='Convex Hull'
        # )
    
    # Calculate AUC
    # auc = calculate_auc(optimal_points[x_col].values, optimal_points[y_col].values)
    
    # Add model/agent names as labels with better styling
    for _, row in pareto_df.iterrows():
        color = '#e74c3c' if row['pareto_optimal'] else '#3498db'
        weight = 'bold' if row['pareto_optimal'] else 'normal'
        ax.annotate(
            row[model_col],
            (row[x_col], row[y_col]),
            xytext=(7, 7),
            textcoords='offset points',
            fontsize=10,
            color=color,
            weight=weight,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc='white',
                ec=color,
                alpha=0.7
            )
        )
    
    # Add title and labels with better styling
    ax.set_title(title, fontsize=16, pad=20, weight='bold')
    ax.set_xlabel(x_label, fontsize=14, labelpad=10)
    ax.set_ylabel(y_label, fontsize=14, labelpad=10)
    
    # Improve grid appearance
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Add a legend with better styling
    legend = ax.legend(
        loc='best',
        frameon=True,
        framealpha=0.95,
        facecolor='white',
        edgecolor='lightgray',
        fontsize=12
    )
    
    # Add a border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{filename}', dpi=300, bbox_inches='tight')
    print(f"Saved file: visualizations/{filename}")
    
    # Save AUC to CSV
    # csv_filename = f'visualizations/auc_data/{filename.replace(".png", "")}_auc.csv'
    
    # with open(csv_filename, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Benchmark', 'AUC'])
    #     writer.writerow([title, auc])
    
    # print(f"Saved AUC data: {csv_filename}")
    
    # Return the Pareto dataframe with AUC
    # pareto_df['auc'] = auc
    # return pareto_df

def grid_pareto_frontier_by_benchmark(tasks, merged_df, x_col, y_col, x_label, y_label, 
                                     num_cols, filename, minimize_x=True, maximize_y=True, 
                                     model_col='model_name_short'):
    """
    Create Pareto frontier plots in a grid for each benchmark
    
    Args:
        tasks: names of benchmarks to group by
        merged_df: dataframe with data to plot
        x_col: Column name for x metric (e.g., latency, cost)
        y_col: Column name for y metric (e.g., win_rate, accuracy)
        x_label: x-axis label
        y_label: y-axis label
        num_cols: number of columns in grid
        filename: final plot name
        minimize_x: Whether to minimize the x metric (True for cost/latency)
        maximize_y: Whether to maximize the y metric (True for win_rate/accuracy)
        model_col: Column name for model/agent names
    """
    # Ensure directories exist
    os.makedirs('visualizations/new_plots', exist_ok=True)
    os.makedirs('visualizations/auc_data', exist_ok=True)
    
    # Calculate the number of tasks
    n_tasks = len(tasks)
    
    # If only one task or no tasks, create a single plot
    if n_tasks <= 1:
        if n_tasks == 1:
            task = tasks[0]
            df_t = merged_df[merged_df['benchmark_name'] == task]
            title = f"Pareto Frontier: {task} - {y_label} vs. {x_label}"
        else:
            # If no tasks, use all data
            df_t = merged_df
            title = f"Pareto Frontier: All Benchmarks - {y_label} vs. {x_label}"
        
        # Use the existing plot_pareto_frontier function for a single plot
        pareto_df = plot_pareto_frontier(
            df_t, x_col, y_col, title, x_label, y_label, 
            filename, minimize_x, maximize_y, model_col
        )
        
        return pareto_df
    
    # Multiple tasks - create a grid of subplots
    n_cols = min(num_cols, n_tasks)  # Maximum number of columns
    n_rows = (n_tasks + n_cols - 1) // n_cols  
    
    # Create a single figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), facecolor='white')
    
    # Add a title to the entire figure
    fig.suptitle(f"Pareto Frontiers: {y_label} vs. {x_label} by Benchmark", 
                 fontsize=18, weight='bold', y=0.98)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    all_pareto_dfs = []
    auc_data = []  # Store benchmark names and AUC values
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        df_t = merged_df[merged_df['benchmark_name'] == task]
        benchmark_name = task
        
        # Skip if no data for this benchmark
        if len(df_t) == 0:
            ax.set_visible(False)
            continue
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
        
        # Identify Pareto optimal points for this benchmark
        pareto_df = identify_pareto_optimal(df_t, x_col, y_col, minimize_x, maximize_y)
        
        # Plot non-optimal points first
        non_optimal = pareto_df[~pareto_df['pareto_optimal']]
        optimal = pareto_df[pareto_df['pareto_optimal']]
        
        # Plot non-optimal points
        ax.scatter(
            non_optimal[x_col], 
            non_optimal[y_col], 
            s=100,
            color='#3498db',
            alpha=0.7,
            edgecolor='white',
            linewidth=1,
            label='Non-Optimal'
        )
        
        # Plot optimal points
        ax.scatter(
            optimal[x_col], 
            optimal[y_col], 
            s=120,
            color='#e74c3c',
            alpha=0.9,
            edgecolor='white',
            linewidth=1.5,
            marker='D',
            label='Pareto Optimal'
        )
        
        # Connect Pareto optimal points with a line
        optimal_points = optimal.sort_values(x_col)
        ax.plot(
            optimal_points[x_col], 
            optimal_points[y_col], 
            color='#e74c3c', 
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
        
        # Plot convex hull if there are enough points
        # if len(optimal_points) >= 3:
        #     # Get coordinates for convex hull
        #     points = np.column_stack([optimal_points[x_col].values, optimal_points[y_col].values])
        #     hull = ConvexHull(points)
            
        #     # Get hull vertices in order
        #     hull_vertices = []
        #     for vertex in hull.vertices:
        #         hull_vertices.append(points[vertex])
        #     hull_vertices = np.array(hull_vertices)
            
            # # Plot the convex hull
            # ax.plot(
            #     hull_vertices[:, 0],
            #     hull_vertices[:, 1],
            #     color='#2ecc71',  # Green color for convex hull
            #     linestyle='-',
            #     linewidth=1.5,
            #     alpha=0.7,
            #     label='Convex Hull'
            # )
        
        # Add model/agent names as labels with better styling
        for _, row in pareto_df.iterrows():
            color = '#e74c3c' if row['pareto_optimal'] else '#3498db'
            weight = 'bold' if row['pareto_optimal'] else 'normal'
            fontsize = 9 if row['pareto_optimal'] else 8
            
            # Only add box around Pareto optimal points to reduce clutter
            if row['pareto_optimal']:
                bbox_props = dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec=color,
                    alpha=0.7
                )
            else:
                bbox_props = None
                
            ax.annotate(
                row[model_col],
                (row[x_col], row[y_col]),
                xytext=(7, 7),
                textcoords='offset points',
                fontsize=fontsize,
                color=color,
                weight=weight,
                bbox=bbox_props
            )
        
        # Add title and labels with better styling
        ax.set_title(benchmark_name, fontsize=14, pad=10, weight='bold')
        ax.set_xlabel(x_label, fontsize=12, labelpad=8)
        ax.set_ylabel(y_label, fontsize=12, labelpad=8)
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Add a subtle border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgray')
            spine.set_linewidth(0.5)
        
        # # Store the pareto dataframe with AUC
        # pareto_df['auc'] = auc
        # all_pareto_dfs.append(pareto_df)
    
    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles, 
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,  # Include Convex Hull in legend
            frameon=True,
            framealpha=0.95,
            facecolor='white',
            edgecolor='lightgray',
            fontsize=12
        )
    
    # Hide any unused subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for annotations and title
    plt.savefig(f'visualizations/new_plots/convex_{filename}', dpi=300, bbox_inches='tight')
    print(f"Saved file: visualizations/new_plots/convex_{filename}")
    
    # Combine all pareto dataframes
    if all_pareto_dfs:
        combined_pareto_df = pd.concat(all_pareto_dfs, ignore_index=True)
        return combined_pareto_df
    else:
        return None


def calculate_pareto_distance(df, x_col, y_col, minimize_x=True, maximize_y=True):
#     """
#     Calculate the distance of each point to the Pareto frontier.
    
#     Args:
#         df: DataFrame with Pareto optimal flags
#         x_col: Column name for x metric (e.g., latency, cost)
#         y_col: Column name for y metric (e.g., win_rate, accuracy)
#         minimize_x: Whether to minimize the x metric (True for cost/latency)
#         maximize_y: Whether to maximize the y metric (True for win_rate/accuracy)
        
#     Returns:
#         DataFrame with distance to Pareto frontier
#     """
#     df = df.copy()
    
#     # For points already on the Pareto frontier, distance is 0
#     df['pareto_distance'] = 0.0
    
#     # Get Pareto optimal points
#     pareto_points = df[df['pareto_optimal']].copy()
#     non_pareto_points = df[~df['pareto_optimal']].copy()
    
#     if len(pareto_points) < 2 or len(non_pareto_points) == 0:
#         return df
    
#     # Sort Pareto points by x
#     pareto_points = pareto_points.sort_values(by=x_col)
    
#     # Get coordinates of Pareto points
#     pareto_x = pareto_points[x_col].values
#     pareto_y = pareto_points[y_col].values
    
#     # For each non-Pareto point, calculate distance to Pareto frontier
#     for idx, row in non_pareto_points.iterrows():
#         point_x = row[x_col]
#         point_y = row[y_col]
        
#         min_distance = float('inf')
        
          ##########################################
          ####### DISTANCE CALCULATION HERE #########
          ##########################################

#         df.loc[idx, 'pareto_distance'] = min_distance
    
    # return df
    return None

# def save_pareto_distances(merged_df, tasks, x_col, y_col, model_col='model_name_short', 
#                           minimize_x=True, maximize_y=True, filename='pareto_distances.csv'):
#     """
#     Calculate and save the distance of each model from the Pareto frontier for each task.
#     """
#     # Ensure directory exists
#     os.makedirs('visualizations/pareto_distances', exist_ok=True)
    
#     all_distances = []
    
#     # Process each task
#     for task in tasks:
#         # Filter data for this task
#         df_task = merged_df[merged_df['benchmark_name'] == task].copy()
        
#         if len(df_task) < 2:
#             continue
        
#         # Identify Pareto optimal points
#         pareto_df = identify_pareto_optimal(df_task, x_col, y_col, minimize_x, maximize_y)
        
#         # Calculate distances
#         distance_df = calculate_pareto_distance(pareto_df, x_col, y_col, minimize_x, maximize_y)
        
#         # Add task name
#         distance_df['benchmark_name'] = task
        
#         # Select relevant columns
#         result_df = distance_df[[model_col, 'benchmark_name', x_col, y_col, 'pareto_optimal', 'pareto_distance']]
        
#         all_distances.append(result_df)
    
#     if not all_distances:
#         print("No data available for calculating Pareto distances")
#         return None
    
#     # Combine results from all tasks
#     combined_df = pd.concat(all_distances, ignore_index=True)
    
#     # Save to CSV
#     csv_path = f'visualizations/pareto_distances/{filename}'
#     combined_df.to_csv(csv_path, index=False)
#     print(f"Saved Pareto distances to: {csv_path}")

def get_max_accuracy():
    full_dataset = pd.read_csv('cleaned_all_metrics.csv')
    model_accuracy = full_dataset.loc[full_dataset.groupby(['model_name_short', 'benchmark_name'])['accuracy'].idxmax()].reset_index(drop=True)
    model_accuracy = model_accuracy[['benchmark_name', 'model_name_short', 'accuracy', 'total_cost', 'mean_latency']]
    # print(model_accuracy.head(), model_accuracy.columns)
    return model_accuracy

def get_mean_latency():
    full_dataset = pd.read_csv('cleaned_all_metrics.csv')
    model_latency = full_dataset.groupby(['model_name_short', 'benchmark_name'])[['mean_latency']].mean()

    # mean latency of models across benchmarks
    model_mean_latency = model_latency.groupby('model_name_short')['mean_latency'].mean().reset_index()
    model_mean_latency = model_mean_latency.rename(columns={'mean_latency':'mean_of_mean_latency'})
    model_mean_latency.to_csv('data/model_mean_latency.csv')

def get_mean_cost():
    full_dataset = pd.read_csv('cleaned_all_metrics.csv')
    model_cost = full_dataset.groupby(['model_name_short', 'benchmark_name'])[['total_cost']].mean()

    # mean latency of models across benchmarks
    model_mean_cost = model_cost.groupby('model_name_short')['total_cost'].mean().reset_index()
    model_mean_cost = model_mean_cost.rename(columns={'total_cost':'mean_cost'})
    model_mean_cost.to_csv('data/model_mean_cost.csv')

def cost_accuracy():
    # model_costs = pd.read_csv('model_total_usage.csv')
    # model_accuracy = pd.read_csv('model_accuracy.csv')
    # df_m = model_accuracy.merge(model_costs, on=['model_name_short', 'benchmark_name'], how='left')
    df_m = get_max_accuracy()
    tasks = df_m['benchmark_name'].unique()
    grid_pareto_frontier_by_benchmark(tasks, df_m, 'total_cost', 'accuracy', 'Total Cost', 'Accuracy', 5, 'model_cost_accuracy.png')
    # save_pareto_distances(df_m, tasks, 'total_cost', 'accuracy')

def latency_accuracy():
    # model_latency = pd.read_csv('model_latency.csv')
    # model_accuracy = pd.read_csv('model_accuracy.csv')
    df_m = get_max_accuracy()
    # df_m = acc.merge(latency, on=['model_name_short', 'benchmark_name'], how='left')
    tasks = df_m['benchmark_name'].unique()
    grid_pareto_frontier_by_benchmark(tasks, df_m, 'mean_latency', 'accuracy', 'Latency', 'Accuracy', 5, 'model_latency_accuracy.png', )

def cost_win_rate():
    # with model win rates and data/model_mean_cost, plot using plot_pareto_fronteir function
    get_mean_cost()
    model_mean_costs = pd.read_csv('data/model_mean_cost.csv')
    model_win_rates_max = pd.read_csv('model_win_rates_max.csv')
    model_win_rates_pareto = pd.read_csv('model_win_rates_pareto.csv')

    # plot pareto frontier for win rate calculation using max accuracy
    df_m = pd.merge(model_mean_costs, model_win_rates_max, on='model_name_short', how='inner')
    cols = ['model_name_short', 'mean_cost', 'win_rate_mean', 'overall_win_rate']
    df_m = df_m[cols].copy()
    plot_pareto_frontier(df_m, 'mean_cost', 'overall_win_rate', 'Max Accuracy Win Rate vs. Mean Cost', 'Mean Cost', 'Win Rate', 'new_plots/cost_win_rate_max.png')

    # plot pareto frontier for win rate calculation using distance from convex hull
    df_m = pd.merge(model_mean_costs, model_win_rates_pareto, on='model_name_short', how='inner')
    cols = ['model_name_short', 'mean_cost', 'win_rate_mean', 'overall_win_rate']
    df_m = df_m[cols].copy()
    plot_pareto_frontier(df_m, 'mean_cost', 'overall_win_rate', 'Distance from Convex Hull Win Rate vs. Mean Cost', 'Mean Cost', 'Win Rate', 'new_plots/cost_win_rate_pareto.png')

def latency_win_rate():
    # with model win rates and data/model_mean_latency, plot using plot_pareto_fronteir function
    get_mean_latency()
    model_mean_latencies = pd.read_csv('data/model_mean_latency.csv')
    model_win_rates_max = pd.read_csv('model_win_rates_max.csv')
    model_win_rates_pareto = pd.read_csv('model_win_rates_pareto.csv')

    # plot pareto frontier for win rate calculation using max accuracy
    df_m = pd.merge(model_mean_latencies, model_win_rates_max, on='model_name_short', how='inner')
    cols = ['model_name_short', 'mean_of_mean_latency', 'win_rate_mean', 'overall_win_rate']
    df_m = df_m[cols].copy()
    plot_pareto_frontier(df_m, 'mean_of_mean_latency', 'overall_win_rate', 'Max Accuracy Win Rate vs. Mean Latency', 'Mean Latency', 'Win Rate', 'new_plots/latency_win_rate_max.png')

    # plot pareto frontier for win rate calculation using distance from convex hull
    df_m = pd.merge(model_mean_latencies, model_win_rates_pareto, on='model_name_short', how='inner')
    cols = ['model_name_short', 'mean_of_mean_latency', 'win_rate_mean', 'overall_win_rate']
    df_m = df_m[cols].copy()
    plot_pareto_frontier(df_m, 'mean_of_mean_latency', 'overall_win_rate', 'Distance from Convex Hull Win Rate vs. Mean Latency', 'Mean Latency', 'Win Rate', 'new_plots/latency_win_rate_pareto.png')

cost_accuracy()
latency_accuracy()
cost_win_rate()
latency_win_rate()

# minimize_x=True, maximize_y=True, 
#                                      model_col='model_name_short'