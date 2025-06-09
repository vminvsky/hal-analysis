import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.spatial import ConvexHull
import os
import csv

def calculate_auc(x, y):
    """
    Calculate the area under the curve (AUC) using the trapezoidal rule.
    
    Args:
        x: x-coordinates
        y: y-coordinates
        
    Returns:
        Area under the curve
    """
    # Sort points by x
    indices = np.argsort(x)
    x_sorted = np.array(x)[indices]
    y_sorted = np.array(y)[indices]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapezoid(y_sorted, x_sorted)
    return auc

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
    
    # Sort by x (ascending if minimize_x, descending otherwise) 
    # and y (descending if maximize_y, ascending otherwise)
    x_ascending = minimize_x
    y_ascending = not maximize_y
    
    df = df.sort_values([x_col, y_col], ascending=[x_ascending, y_ascending])
    
    # Initialize Pareto frontier with the first point
    pareto_frontier = [df.iloc[0]]
    
    # Iterate through remaining points
    for i in range(1, len(df)):
        current_point = df.iloc[i]
        last_pareto = pareto_frontier[-1]
        
        # If current point has better y than the last Pareto optimal point,
        # add it to the Pareto frontier
        if (maximize_y and current_point[y_col] > last_pareto[y_col]) or \
           (not maximize_y and current_point[y_col] < last_pareto[y_col]):
            pareto_frontier.append(current_point)
    # print("pareto frontier points", pareto_frontier)
    pareto_df = pd.DataFrame(pareto_frontier)
    df['pareto_optimal'] = df.index.isin(pareto_df.index)
    non_optimal = df[~df['pareto_optimal']]
    optimal = df[df['pareto_optimal']]
    optimal_points = optimal.sort_values(x_col)

    # Get the convex hull if there are enough points
    if len(optimal_points) >= 3:
        # Get coordinates for convex hull
        points = np.column_stack([optimal_points[x_col].values, optimal_points[y_col].values])
        hull = ConvexHull(points)
            
        # Get hull vertices in order
        hull_vertices = []
        for vertex in hull.vertices:
            hull_vertices.append(points[vertex])
        hull_vertices = np.array(hull_vertices)
        hull_set = set(map(tuple, hull_vertices))
        df['pareto_optimal'] = df.apply(lambda row: (row[x_col], row[y_col]) in hull_set, axis=1)
    else:
        df['pareto_optimal'] = df.index.isin(pareto_df.index)

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
    
    # Calculate AUC
    auc = calculate_auc(optimal_points[x_col].values, optimal_points[y_col].values)
    
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
    csv_filename = f'visualizations/auc_data/{filename.replace(".png", "")}_auc.csv'
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Benchmark', 'AUC'])
        writer.writerow([title, auc])
    
    print(f"Saved AUC data: {csv_filename}")
    
    # Return the Pareto dataframe with AUC
    pareto_df['auc'] = auc
    return pareto_df

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
    
    #all_pareto_dfs = []
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
        
        auc = calculate_auc(optimal_points[x_col].values, optimal_points[y_col].values)
        auc_data.append((benchmark_name, auc))

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
        
        # Add a border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgray')
            spine.set_linewidth(0.5)
        
        # Store the pareto dataframe with AUC
        pareto_df['auc'] = auc
        # all_pareto_dfs.append(pareto_df)
        # print(pareto_df.head())
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
    plt.savefig(f'visualizations/new_plots/{filename}', dpi=300, bbox_inches='tight')
    print(f"Saved file: visualizations/new_plots/{filename}")
    
    # Save AUCs to CSV
    if auc_data:
        csv_filename = f'visualizations/auc_data/{filename.replace(".png", "")}_auc.csv'
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Benchmark', 'AUC'])
            for benchmark, auc in auc_data:
                writer.writerow([benchmark, auc])
        
        print(f"Saved AUC data: {csv_filename}")
        
        # Create AUC visualization
        create_auc_visualization(auc_data, filename.replace(".png", ""))

def create_auc_visualization(auc_data, base_filename):
    """
    Create a bar chart visualization of AUCs across benchmarks.
    
    Args:
        auc_data: List of tuples (benchmark_name, auc_value)
        base_filename: Base filename for saving the visualization
    """
    # Extract benchmark names and AUC values
    benchmarks = [item[0] for item in auc_data]
    auc_values = [item[1] for item in auc_data]
    
    if not benchmarks:
        print("No AUC data available for visualization")
        return
    
    # Create bar chart
    plt.figure(figsize=(12, 8), facecolor='white')
    
    # Set a more attractive style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create bars with gradient color
    bars = plt.bar(benchmarks, auc_values, color='#3498db', alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01 * max(auc_values),
            f'{height:.4f}',
            ha='center', 
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add title and labels
    plt.title('Area Under Curve (AUC) by Benchmark', fontsize=16, pad=20, weight='bold')
    plt.xlabel('Benchmark', fontsize=14, labelpad=10)
    plt.ylabel('AUC', fontsize=14, labelpad=10)
    
    # Improve grid appearance
    plt.grid(True, linestyle='--', alpha=0.3, color='gray', axis='y')
    
    # Rotate x-axis labels if there are many benchmarks
    if len(benchmarks) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('visualizations/auc_visualizations', exist_ok=True)
    plt.savefig(f'visualizations/auc_visualizations/{base_filename}_auc_viz.png', dpi=300, bbox_inches='tight')
    print(f"Saved AUC visualization: visualizations/auc_visualizations/{base_filename}_auc_viz.png")

def inverse_cost_accuracy():
    model_costs = pd.read_csv('model_total_usage.csv')
    model_accuracy = pd.read_csv('model_accuracy.csv')
    df_m = model_accuracy.merge(model_costs, on=['model_name_short', 'benchmark_name'], how='left')
    df_m['inverse_cost'] = df_m['total_cost'].apply(lambda x: 1/x if x != 0 else np.nan)
    tasks = df_m['benchmark_name'].unique()
    grid_pareto_frontier_by_benchmark(tasks, df_m, 'inverse_cost', 'accuracy', 'Inverse Cost', 'Accuracy', 4, 'inverse_cost_accuracy.png')

inverse_cost_accuracy()
