import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataloaders.abstract import DataCombiner

def calculate_pareto_frontier(df, group_by_cols=['benchmark_name', 'agent_name_short']):
    """
    Calculate average accuracy and cost for each model across different benchmarks and agent scaffolds.
    
    Args:
        df: DataFrame containing the results
        group_by_cols: Columns to group by (control variables)
        
    Returns:
        DataFrame with accuracy and cost metrics for each model
    """
    # Create a results container
    all_results = []
    
    # Group by the control variables
    for name, group in df.groupby(group_by_cols):
        # For each group (benchmark+agent combination), calculate metrics for each model
        models = group['model_name_short'].unique()
        
        for model in models:
            model_data = group[group['model_name_short'] == model]
            
            # Get accuracy and cost
            accuracy = model_data['accuracy'].values[0]
            total_cost = model_data['total_cost'].values[0]
            
            # Store the result
            result = {
                'model_name_short': model,
                'accuracy': accuracy,
                'total_cost': total_cost
            }
            
            # Add the group information
            for i, col in enumerate(group_by_cols):
                result[col] = name[i] if isinstance(name, tuple) else name
                
            all_results.append(result)
    
    return pd.DataFrame(all_results)

def aggregate_metrics(metrics_df, group_by='model_name_short'):
    """
    Aggregate metrics by model or other grouping variables.
    
    Args:
        metrics_df: DataFrame with metrics
        group_by: Column or list of columns to group by for aggregation
        
    Returns:
        DataFrame with aggregated metrics
    """
    # Group by the specified column(s) and calculate average metrics
    agg_df = metrics_df.groupby(group_by).agg({
        'accuracy': ['mean', 'std', 'count'],
        'total_cost': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Flatten the column names
    agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns]
    
    return agg_df

def identify_pareto_optimal(df, x_col='total_cost_mean', y_col='accuracy_mean'):
    """
    Identify Pareto optimal models (best accuracy for a given cost).
    
    Args:
        df: DataFrame with aggregated metrics
        x_col: Column name for cost metric
        y_col: Column name for accuracy metric
        
    Returns:
        DataFrame with Pareto optimal flag
    """
    df = df.copy()
    
    # Sort by cost (ascending) and accuracy (descending)
    df = df.sort_values([x_col, y_col], ascending=[True, False])
    
    # Initialize Pareto frontier with the first model (lowest cost)
    pareto_frontier = [df.iloc[0]]
    
    # Iterate through remaining models
    for i in range(1, len(df)):
        current_model = df.iloc[i]
        last_pareto = pareto_frontier[-1]
        
        # If current model has higher accuracy than the last Pareto optimal model,
        # add it to the Pareto frontier
        if current_model[y_col] > last_pareto[y_col]:
            pareto_frontier.append(current_model)
    
    # Create a DataFrame from the Pareto frontier
    pareto_df = pd.DataFrame(pareto_frontier)
    
    # Add a flag to the original DataFrame indicating Pareto optimal models
    df['pareto_optimal'] = df.index.isin(pareto_df.index)
    
    return df

def plot_pareto_frontier(df, x_col='total_cost_mean', y_col='accuracy_mean', 
                         title="Pareto Frontier: Accuracy vs. Cost"):
    """
    Plot the Pareto frontier for accuracy vs. cost.
    
    Args:
        df: DataFrame with metrics and Pareto optimal flag
        x_col: Column name for cost metric
        y_col: Column name for accuracy metric
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot all models
    sns.scatterplot(
        x=x_col, 
        y=y_col, 
        hue='pareto_optimal',
        style='pareto_optimal',
        s=100,
        data=df
    )
    
    # Connect Pareto optimal points with a line
    pareto_df = df[df['pareto_optimal']].sort_values(x_col)
    plt.plot(pareto_df[x_col], pareto_df[y_col], 'r--')
    
    # Add model names as labels
    for _, row in df.iterrows():
        plt.annotate(
            row['model_name_short'],
            (row[x_col], row[y_col]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title(title)
    plt.xlabel('Average Cost')
    plt.ylabel('Average Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt

def main():
    # Load data for all tasks you want to analyze
    tasks = ['taubench_retail', 'usaco', 'test', 'taubench', 'swebench', 'react', 
             'planexec', 'ipfuncall', 'inspect', 'gaia', 'fullcode', 'cybench', 
             'agentharm_', 'agentharm_benign']
    
    all_data = []
    for task in tasks:
        try:
            combiner = DataCombiner(task)
            df = combiner.load()
            all_data.append(df)
        except Exception as e:
            print(f"Error loading task {task}: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Calculate metrics for each model, controlling for benchmark and agent
        metrics_df = calculate_pareto_frontier(combined_df)
        
        # Save the raw metrics
        metrics_df.to_csv('model_metrics_by_benchmark_agent.csv', index=False)
        
        # Aggregate by model
        model_metrics = aggregate_metrics(metrics_df)
        
        # Identify Pareto optimal models
        pareto_df = identify_pareto_optimal(model_metrics)
        
        # Save the aggregated metrics with Pareto optimal flag
        pareto_df.to_csv('model_pareto_analysis.csv', index=False)
        
        # Plot the Pareto frontier
        plt = plot_pareto_frontier(pareto_df)
        plt.savefig('model_pareto_frontier.png')
        
        # Aggregate by benchmark and model
        benchmark_metrics = aggregate_metrics(metrics_df, group_by=['benchmark_name', 'model_name_short'])
        
        # For each benchmark, identify Pareto optimal models
        benchmarks = benchmark_metrics['benchmark_name'].unique()
        all_benchmark_pareto = []
        
        for benchmark in benchmarks:
            benchmark_df = benchmark_metrics[benchmark_metrics['benchmark_name'] == benchmark]
            benchmark_pareto = identify_pareto_optimal(benchmark_df)
            all_benchmark_pareto.append(benchmark_pareto)
            
            # Plot Pareto frontier for this benchmark
            plt = plot_pareto_frontier(
                benchmark_pareto, 
                title=f"Pareto Frontier: {benchmark} - Accuracy vs. Cost"
            )
            plt.savefig(f'pareto_frontier_{benchmark}.png')
        
        # Combine all benchmark Pareto analyses
        all_benchmark_pareto_df = pd.concat(all_benchmark_pareto, ignore_index=True)
        all_benchmark_pareto_df.to_csv('benchmark_pareto_analysis.csv', index=False)
        
        print("Pareto analysis complete. Results saved to CSV files and plots.")
        return combined_df, metrics_df, pareto_df, all_benchmark_pareto_df
    else:
        print("No data loaded.")
        return None, None, None, None

if __name__ == "__main__":
    combined_df, metrics_df, pareto_df, benchmark_pareto_df = main()
