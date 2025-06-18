import pandas as pd
from pareto_utils import (
    plot_pareto_frontier, 
    grid_pareto_frontier_by_benchmark,
    get_max_accuracy,
    get_mean_latency,
    get_mean_cost )


def latency_accuracy():
    df_m = get_max_accuracy()
    tasks = df_m['benchmark_name'].unique()
    grid_pareto_frontier_by_benchmark(tasks, df_m, 'mean_latency', 'accuracy', 'Latency', 'Accuracy', 5, 'model_latency_accuracy.png')


def cost_win_rate():
    # Generate mean cost data
    get_mean_cost()
    model_mean_costs = pd.read_csv('data/model_mean_cost.csv')
    model_win_rates_max = pd.read_csv('data/model_win_rates_max.csv')
    model_win_rates_pareto = pd.read_csv('data/model_win_rates_pareto.csv')

    # Plot pareto frontier for win rate calculation using max accuracy
    df_m = pd.merge(model_mean_costs, model_win_rates_max, on='model_name_short', how='inner')
    cols = ['model_name_short', 'mean_cost', 'win_rate_mean', 'overall_win_rate']
    df_m = df_m[cols].copy()
    plot_pareto_frontier(df_m, 'mean_cost', 'overall_win_rate', 'Max Accuracy Win Rate vs. Mean Cost', 'Mean Cost', 'Win Rate', 'plots/cost_win_rate_max.png')

    # Plot pareto frontier for win rate calculation using distance from convex hull
    df_m = pd.merge(model_mean_costs, model_win_rates_pareto, on='model_name_short', how='inner')
    cols = ['model_name_short', 'mean_cost', 'win_rate_mean', 'overall_win_rate']
    df_m = df_m[cols].copy()
    plot_pareto_frontier(df_m, 'mean_cost', 'overall_win_rate', 'Distance from Convex Hull Win Rate vs. Mean Cost', 'Mean Cost', 'Win Rate', 'plots/cost_win_rate_pareto.png')


def latency_win_rate():
    # Generate mean latency data
    get_mean_latency()
    model_mean_latencies = pd.read_csv('data/model_mean_latency.csv')
    model_win_rates_max = pd.read_csv('data/model_win_rates_max.csv')
    model_win_rates_pareto = pd.read_csv('data/model_win_rates_pareto.csv')

    # Plot pareto frontier for win rate calculation using max accuracy
    df_m = pd.merge(model_mean_latencies, model_win_rates_max, on='model_name_short', how='inner')
    cols = ['model_name_short', 'mean_of_mean_latency', 'win_rate_mean', 'overall_win_rate']
    df_m = df_m[cols].copy()
    plot_pareto_frontier(df_m, 'mean_of_mean_latency', 'overall_win_rate', 
        'Max Accuracy Win Rate vs. Mean Latency', 'Mean Latency', 'Win Rate', 
        'plots/latency_win_rate_max.png')

    # Plot pareto frontier for win rate calculation using distance from convex hull
    df_m = pd.merge(model_mean_latencies, model_win_rates_pareto, on='model_name_short', how='inner')
    cols = ['model_name_short', 'mean_of_mean_latency', 'win_rate_mean', 'overall_win_rate']
    df_m = df_m[cols].copy()
    plot_pareto_frontier(
        df_m, 'mean_of_mean_latency', 'overall_win_rate','Distance from Convex Hull Win Rate vs. Mean Latency', 
        'Mean Latency', 'Win Rate', 'plots/latency_win_rate_pareto.png')

if __name__ == "__main__":
    latency_accuracy()
    cost_win_rate()
    latency_win_rate()
