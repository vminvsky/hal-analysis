import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataloaders.abstract import DataCombiner

def main():
    # Load data for all tasks you want to analyze
    # tasks = ['taubench_retail', 'usaco', 'test','taubench', 'swebench', 'react', 'planexec', 'ipfuncall', 'inspect', 'gaia', 'fullcode', 'cybench', 'agentharm_', 'agentharm_benign']  # Add your task names here
    tasks = ['gaia', 'cybench', 'taubench', 'taubench_retail' , 'agentharm_']
    cols = ['model_name_short', 'total_cost', 'benchmark_name', 'agent_name']
    all_data = []
    all_data = pd.DataFrame()
    for task in tasks:
        try:
            combiner = DataCombiner(task)
            df = combiner.load()
            df = df[cols]
            print(df.head())
            all_data = pd.concat([all_data, df])
        except Exception as e:
            print(f"Error loading task {task}: {e}")

    df = all_data
    print(df.keys())
    model_total_usage = df.groupby(['model_name_short', 'benchmark_name'])[['total_cost']].mean()
    benchmark_total_usage = df.groupby(['agent_name', 'benchmark_name'])[['total_cost']].mean()
    model_total_usage.to_csv('model_total_usage.csv')
    benchmark_total_usage.to_csv('benchmark_total_usage.csv')

if __name__ == "__main__":
    main()
