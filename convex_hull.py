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
    
    # Create a DataFrame from the Pareto frontier
    pareto_df = pd.DataFrame(pareto_frontier)
    
    # Add a flag to the original DataFrame indicating Pareto optimal points
    df['pareto_optimal'] = df.index.isin(pareto_df.index)
    
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
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
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
    if len(optimal_points) >= 3:
        # Get coordinates for convex hull
        points = np.column_stack([optimal_points[x_col].values, optimal_points[y_col].values])
        hull = ConvexHull(points)
        
        # Get hull vertices in order
        hull_vertices = []
        for vertex in hull.vertices:
            hull_vertices.append(points[vertex])
        hull_vertices.append(hull_vertices[0])  # Close the loop
        hull_vertices = np.array(hull_vertices)
        
        # Plot the convex hull
        ax.plot(
            hull_vertices[:, 0],
            hull_vertices[:, 1],
            color='#2ecc71',  # Green color for convex hull
            linestyle='-',
            linewidth=2,
            alpha=0.7,
            label='Convex Hull'
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
    
    # Add a subtle border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(0.5)
    
    # Add AUC to the plot
    ax.text(
        0.05, 0.05, 
        f'AUC: {auc:.4f}', 
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc='white',
            ec='gray',
            alpha=0.8
        )
    )
    
    plt.tight_layout()
    
    # Save with higher quality
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
        if len(optimal_points) >= 3:
            # Get coordinates for convex hull
            points = np.column_stack([optimal_points[x_col].values, optimal_points[y_col].values])
            hull = ConvexHull(points)
            
            # Get hull vertices in order
            hull_vertices = []
            for vertex in hull.vertices:
                hull_vertices.append(points[vertex])
            hull_vertices.append(hull_vertices[0])  # Close the loop
            hull_vertices = np.array(hull_vertices)
            
            # Plot the convex hull
            ax.plot(
                hull_vertices[:, 0],
                hull_vertices[:, 1],
                color='#2ecc71',  # Green color for convex hull
                linestyle='-',
                linewidth=1.5,
                alpha=0.7,
                label='Convex Hull'
            )
        
        # Calculate AUC
        auc = calculate_auc(optimal_points[x_col].values, optimal_points[y_col].values)
        auc_data.append((benchmark_name, auc))
        
        # Add AUC to the plot
        ax.text(
            0.05, 0.05, 
            f'AUC: {auc:.4f}', 
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc='white',
                ec='gray',
                alpha=0.8
            )
        )
        
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
        
        # Store the pareto dataframe with AUC
        pareto_df['auc'] = auc
        all_pareto_dfs.append(pareto_df)
    
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
    
    # Combine all pareto dataframes
    if all_pareto_dfs:
        combined_pareto_df = pd.concat(all_pareto_dfs, ignore_index=True)
        return combined_pareto_df
    else:
        return None

def calculate_pareto_distance(df, x_col, y_col, minimize_x=True, maximize_y=True):
    """
    Calculate the distance of each point to the Pareto frontier.
    
    Args:
        df: DataFrame with Pareto optimal flags
        x_col: Column name for x metric (e.g., latency, cost)
        y_col: Column name for y metric (e.g., win_rate, accuracy)
        minimize_x: Whether to minimize the x metric (True for cost/latency)
        maximize_y: Whether to maximize the y metric (True for win_rate/accuracy)
        
    Returns:
        DataFrame with distance to Pareto frontier
    """
    df = df.copy()
    
    # For points already on the Pareto frontier, distance is 0
    df['pareto_distance'] = 0.0
    
    # Get Pareto optimal points
    pareto_points = df[df['pareto_optimal']].copy()
    non_pareto_points = df[~df['pareto_optimal']].copy()
    
    if len(pareto_points) < 2 or len(non_pareto_points) == 0:
        return df
    
    # Sort Pareto points by x
    pareto_points = pareto_points.sort_values(by=x_col)
    
    # Get coordinates of Pareto points
    pareto_x = pareto_points[x_col].values
    pareto_y = pareto_points[y_col].values
    
    # For each non-Pareto point, calculate distance to Pareto frontier
    for idx, row in non_pareto_points.iterrows():
        point_x = row[x_col]
        point_y = row[y_col]
        
        # Initialize with a large value
        min_distance = float('inf')
        
        # Check if point is outside the x-range of Pareto frontier
        if point_x < pareto_x[0]:
            # Point is to the left of Pareto frontier
            # Calculate distance to the leftmost Pareto point
            dist = np.sqrt((point_x - pareto_x[0])**2 + (point_y - pareto_y[0])**2)
            min_distance = min(min_distance, dist)
        elif point_x > pareto_x[-1]:
            # Point is to the right of Pareto frontier
            # Calculate distance to the rightmost Pareto point
            dist = np.sqrt((point_x - pareto_x[-1])**2 + (point_y - pareto_y[-1])**2)
            min_distance = min(min_distance, dist)
        else:
            # Point is within the x-range of Pareto frontier
            # Calculate distance to each line segment of the Pareto frontier
            for i in range(len(pareto_x) - 1):
                x1, y1 = pareto_x[i], pareto_y[i]
                x2, y2 = pareto_x[i+1], pareto_y[i+1]
                
                # Calculate distance to line segment
                # First, check if the projection of the point onto the line segment falls within the segment
                segment_length_squared = (x2 - x1)**2 + (y2 - y1)**2
                if segment_length_squared == 0:
                    # Degenerate case: segment is a point
                    dist = np.sqrt((point_x - x1)**2 + (point_y - y1)**2)
                else:
                    # Calculate projection
                    t = max(0, min(1, ((point_x - x1) * (x2 - x1) + (point_y - y1) * (y2 - y1)) / segment_length_squared))
                    proj_x = x1 + t * (x2 - x1)
                    proj_y = y1 + t * (y2 - y1)
                    dist = np.sqrt((point_x - proj_x)**2 + (point_y - proj_y)**2)
                
                min_distance = min(min_distance, dist)
        
        # Update distance in the DataFrame
        df.loc[idx, 'pareto_distance'] = min_distance
    
    return df

def save_pareto_distances(merged_df, tasks, x_col, y_col, model_col='model_name_short', 
                          minimize_x=True, maximize_y=True, filename='pareto_distances.csv'):
    """
    Calculate and save the distance of each model from the Pareto frontier for each task.
    
    Args:
        merged_df: DataFrame with data for all tasks
        tasks: List of task/benchmark names
        x_col: Column name for x metric (e.g., latency, cost)
        y_col: Column name for y metric (e.g., win_rate, accuracy)
        model_col: Column name for model/agent names
        minimize_x: Whether to minimize the x metric (True for cost/latency)
        maximize_y: Whether to maximize the y metric (True for win_rate/accuracy)
        filename: Output CSV filename
        
    Returns:
        DataFrame with distances for all tasks
    """
    # Ensure directory exists
    os.makedirs('visualizations/pareto_distances', exist_ok=True)
    
    all_distances = []
    
    # Process each task
    for task in tasks:
        # Filter data for this task
        df_task = merged_df[merged_df['benchmark_name'] == task].copy()
        
        if len(df_task) < 2:
            continue
        
        # Identify Pareto optimal points
        pareto_df = identify_pareto_optimal(df_task, x_col, y_col, minimize_x, maximize_y)
        
        # Calculate distances
        distance_df = calculate_pareto_distance(pareto_df, x_col, y_col, minimize_x, maximize_y)
        
        # Add task name
        distance_df['benchmark_name'] = task
        
        # Select relevant columns
        result_df = distance_df[[model_col, 'benchmark_name', x_col, y_col, 'pareto_optimal', 'pareto_distance']]
        
        all_distances.append(result_df)
    
    if not all_distances:
        print("No data available for calculating Pareto distances")
        return None
    
    # Combine results from all tasks
    combined_df = pd.concat(all_distances, ignore_index=True)
    
    # Save to CSV
    csv_path = f'visualizations/pareto_distances/{filename}'
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved Pareto distances to: {csv_path}")
    
    return combined_df

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

def cost_accuracy():
    model_costs = pd.read_csv('model_total_usage.csv')
    model_accuracy = pd.read_csv('model_accuracy.csv')
    df_m = model_accuracy.merge(model_costs, on=['model_name_short', 'benchmark_name'], how='left')
    tasks = df_m['benchmark_name'].unique()
    grid_pareto_frontier_by_benchmark(tasks, df_m, 'total_cost', 'accuracy', 'Total Cost', 'Accuracy', 4, 'model_cost_accuracy.png')
    save_pareto_distances(df_m, tasks, 'total_cost', 'accuracy')

cost_accuracy()