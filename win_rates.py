import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.dataloaders.abstract import DataCombiner

def calculate_max_win_rates(df, group_by_cols=['benchmark_name'], group_by_cols_2='model_name_short'):
    """
    Calculate win rates for each model across different benchmarks using the maximum accuracy
    each model achieves across all agent scaffolds.
    
    Args:
        df: DataFrame containing the results
        group_by_cols: Columns to group by (control variables)
        group_by_cols_2: Column representing the models to compare
        
    Returns:
        DataFrame with win rates for each model
    """
    
    # Create a results container
    all_results = []
    
    # First, find the maximum accuracy for each model across all agents for each benchmark
    max_accuracies = {}
    
    for benchmark, benchmark_group in df.groupby('benchmark_name'):
        max_accuracies[benchmark] = {}
        
        for model in benchmark_group[group_by_cols_2].unique():
            # Get all accuracies for this model across different agents in this benchmark
            model_accuracies = benchmark_group[benchmark_group[group_by_cols_2] == model]['accuracy']

            # Store the maximum accuracy
            max_accuracies[benchmark][model] = model_accuracies.max()
    
    # Group by the control variables (benchmarks)
    for name, group in df.groupby(group_by_cols):
        benchmark = name if not isinstance(name, tuple) else name[0]
        
        # Get unique models for this benchmark
        models = df[df['benchmark_name'] == benchmark][group_by_cols_2].unique()
        
        for model_a in models:
            wins = 0
            comparisons = 0
            
            for model_b in models:
                if model_a == model_b:
                    continue
                    
                acc_a = max_accuracies[benchmark][model_a]
                acc_b = max_accuracies[benchmark][model_b]
                
                if acc_a > acc_b:
                    wins += 1
                
                comparisons += 1
            
            # Calculate win rate
            win_rate = wins / comparisons if comparisons > 0 else np.nan
            
            # Store the result
            result = {
                group_by_cols_2: model_a,
                'win_rate': win_rate,
                'wins': wins,
                'comparisons': comparisons
            }
            
            # Add the group information
            for i, col in enumerate(group_by_cols):
                result[col] = name[i] if isinstance(name, tuple) else name
                
            all_results.append(result)
    
    return pd.DataFrame(all_results)

def calculate_pareto_win_rates(df, group_by_cols=['benchmark_name'], group_by_cols_2='model_name_short'):
    """
    Calculate win rates for each model across different benchmarks using the distance from the convex hull of the pareto frontier across all agent scaffolds.
    
    Args:
        df: DataFrame containing the results
        group_by_cols: Columns to group by (control variables)
        group_by_cols_2: Column representing the models to compare
        
    Returns:
        DataFrame with win rates for each model
    """

    df = pd.read_csv('visualizations/pareto_distances/pareto_distances.csv')

    all_results = []

    # Group by the control variables (benchmarks)
    for name, group in df.groupby(group_by_cols):
        benchmark = name if not isinstance(name, tuple) else name[0]
        
        # Get unique models for this benchmark
        models = df[df['benchmark_name'] == benchmark][group_by_cols_2].unique()
        
        for model_a in models:
            wins = 0
            comparisons = 0
            
            for model_b in models:
                if model_a == model_b:
                    continue
                    
                dist_a = df.loc[df['model_name_short'] == model_a, 'pareto_distance']
                dist_b = df.loc[df['model_name_short'] == model_b, 'pareto_distance']
                
                if dist_a < dist_b:
                    wins += 1
                
                comparisons += 1
            
            # Calculate win rate
            win_rate = wins / comparisons if comparisons > 0 else np.nan
            
            # Store the result
            result = {
                group_by_cols_2: model_a,
                'win_rate': win_rate,
                'wins': wins,
                'comparisons': comparisons
            }
            
            # Add the group information
            for i, col in enumerate(group_by_cols):
                result[col] = name[i] if isinstance(name, tuple) else name
                
            all_results.append(result)
    
    # return pd.DataFrame(all_results)


# def calculate_win_rates(df, group_by_cols=['benchmark_name', 'agent_name_short'], group_by_cols_2='model_name_short'):
#     """
#     Calculate win rates for each model across different benchmarks and agent scaffolds.
    
#     Args:
#         df: DataFrame containing the results
#         group_by_cols: Columns to group by (control variables)
        
#     Returns:
#         DataFrame with win rates for each model
#     """


#     # Create a results container
#     all_results = []
    
#     # Group by the control variables
#     for name, group in df.groupby(group_by_cols):
#         # For each group (benchmark+agent combination), find all pairwise comparisons
#         models = group[group_by_cols_2].unique()
        
#         for model_a in models:
#             wins = 0
#             comparisons = 0
            
#             for model_b in models:
#                 if model_a == model_b:
#                     continue
                    
#                 acc_a = group[group[group_by_cols_2] == model_a]['accuracy'].values[0]
#                 acc_b = group[group[group_by_cols_2] == model_b]['accuracy'].values[0]
                
#                 if acc_a > acc_b:
#                     wins += 1
                
#                 comparisons += 1
            
#             # Calculate win rate
#             win_rate = wins / comparisons if comparisons > 0 else np.nan
            
#             # Store the result
#             result = {
#                 group_by_cols_2: model_a,
#                 'win_rate': win_rate,
#                 'wins': wins,
#                 'comparisons': comparisons
#             }
            
#             # Add the group information
#             for i, col in enumerate(group_by_cols):
#                 result[col] = name[i] if isinstance(name, tuple) else name
                
#             all_results.append(result)
    
#     return pd.DataFrame(all_results)

def aggregate_win_rates(win_rates_df, group_by='model_name_short'):
    """
    Aggregate win rates by model or other grouping variables.
    
    Args:
        win_rates_df: DataFrame with win rates
        group_by: Column to group by for aggregation
        
    Returns:
        DataFrame with aggregated win rates
    """
    # Group by the specified column and calculate average win rate
    agg_df = win_rates_df.groupby(group_by).agg({
        'win_rate': ['mean', 'std', 'count'],
        'wins': 'sum',
        'comparisons': 'sum'
    }).reset_index()
    
    # Flatten the column names
    agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns]
    
    # Calculate overall win rate
    agg_df['overall_win_rate'] = agg_df['wins_sum'] / agg_df['comparisons_sum']
    
    return agg_df

def plot_win_rates(agg_df, title="Model Win Rates"):
    """
    Plot the win rates for each model.
    
    Args:
        agg_df: Aggregated DataFrame with win rates
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Sort by win rate
    sorted_df = agg_df.sort_values('overall_win_rate', ascending=False)
    
    # Create bar plot
    sns.barplot(x='model_name_short', y='overall_win_rate', data=sorted_df)
    
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt

def main():
    # Load data for all tasks you want to analyze
    tasks = ['taubench_airline', 'colbench_backend_programming', 'colbench_frontend_design', 'gaia', 'scicode', 'scienceagentbench', 'swebench_verified_mini', 'usaco']
    all_data = []
    for task in tqdm(tasks):
        try:
            combiner = DataCombiner(task)
            df = combiner.load()
            all_data.append(df)
        except Exception as e:
            print(f"Error loading task {task}: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Calculate win rates using max accuracy
        win_rates_max = calculate_max_win_rates(combined_df)

        # Calculate win rates using distance from the pareto
        win_rates_pareto = calculate_pareto_win_rates(combined_df)
        
        # Aggregate by model
        model_win_rates_max = aggregate_win_rates(win_rates_max)
        model_win_rates_pareto = aggregate_win_rates(win_rates_pareto)
        
        # Print results
        # print("Overall model win rates:")
        # print(model_win_rates.sort_values('overall_win_rate', ascending=False))
        
        # Plot results
        # plt = plot_win_rates(model_win_rates)
        # plt.savefig('model_win_rates.png')
        
        # You can also analyze by benchmark
        benchmark_win_rates_max = aggregate_win_rates(win_rates_max, group_by=['model_name_short', 'benchmark_name'])
        print("\nModel win rates by benchmark:")
        print(benchmark_win_rates_max.sort_values(['benchmark_name', 'overall_win_rate'], ascending=[True, False]))
        
        # Save results to CSV
        model_win_rates_max.to_csv('model_win_rates_max.csv', index=False)
        benchmark_win_rates_max.to_csv('benchmark_win_rates_max.csv', index=False)

        ##### pareto distance win rates #####

        benchmark_win_rates_pareto = aggregate_win_rates(win_rates_pareto, group_by=['model_name_short', 'benchmark_name'])
        print("\nModel win rates by benchmark:")
        print(benchmark_win_rates_pareto.sort_values(['benchmark_name', 'overall_win_rate'], ascending=[True, False]))
        
        # Save results to CSV
        model_win_rates_pareto.to_csv('model_win_rates_pareto.csv', index=False)
        benchmark_win_rates_pareto.to_csv('benchmark_win_rates_pareto.csv', index=False)
        
        return combined_df, win_rates, model_win_rates, benchmark_win_rates
    else:
        print("No data loaded.")
        return None, None, None, None

if __name__ == "__main__":
    combined_df, win_rates, model_win_rates, benchmark_win_rates = main()
