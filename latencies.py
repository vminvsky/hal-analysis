import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataloaders.abstract import DataCombiner

def list_mean(lst):
    return np.mean(lst) if lst else np.nan

def z_score(lst):
    if len(lst) == 0:
        return np.nan
    if len(lst) == 1:
        return [0.0]
    mean = np.mean(lst)
    std = np.std(lst)
    return [(x - mean) / std for x in lst]

def main():
    Load data for all tasks you want to analyze
    tasks = ['taubench_airline', 'colbench_backend_programming', 'colbench_frontend_design', 'gaia', 'scicode', 'scienceagentbench', 'swebench_verified_mini', 'usaco', 'corebench_hard', 'assistantbench']
    cols = ['model_name_short', 'latencies_per_task', 'benchmark_name', 'agent_name', 'agent_name_short']
    all_data = []
    all_data = pd.DataFrame()
    for task in tasks:
        try:
            combiner = DataCombiner(task)
            df = combiner.load()
            df = df[cols].copy()
            # mean of latencies for all tasks
            df.loc[:, 'latency'] = df['latencies_per_task'].apply(list_mean)
            # df['latency'] = df['latencies_per_task'].apply(list_mean)
            df.loc[:, 'z_scores'] = df['latencies_per_task'].apply(z_score)
            # df['z_scores'] = df['latencies_per_task'].apply(z_score)
            print(df.head())
            all_data = pd.concat([all_data, df])
        except Exception as e:
            print(f"Error loading task {task}: {e}")


    model_latency = df.groupby(['model_name_short', 'benchmark_name'])[['latency']].mean()
    benchmark_latency = df.groupby(['agent_name_short', 'benchmark_name'])[['latency']].mean()
    model_latency.to_csv('model_latency.csv')
    benchmark_latency.to_csv('agent_latency.csv')

    # mean latency of models across benchmarks
    model_mean_latency = model_latency.groupby('model_name_short')['latency'].mean().reset_index()
    model_mean_latency = model_mean_latency.rename(columns={'latency':'mean_latency'})
    model_mean_latency.to_csv('data/model_mean_latency.csv')

    # mean latency of agents across benchmarks
    agent_mean_latency =  benchmark_latency.groupby('agent_name_short')['latency'].mean().reset_index()
    agent_mean_latency = agent_mean_latency.rename(columns={'latency':'mean_latency'})
    agent_mean_latency.to_csv('data/agent_mean_latency.csv')

if __name__ == "__main__":
    main()
