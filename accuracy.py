import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataloaders.abstract import DataCombiner

def main():
    # Load data for all tasks you want to analyze
    tasks = ['taubench_retail', 'usaco', 'test','taubench', 'swebench', 'react', 'planexec', 'ipfuncall', 'inspect', 'gaia', 'fullcode', 'cybench', 'agentharm_', 'agentharm_benign']  # Add your task names here
    # tasks = ['gaia', 'cybench', 'taubench', 'taubench_retail' , 'agentharm_']
    cols = ['model_name_short', 'accuracy', 'benchmark_name', 'agent_name']
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
    model_accuracy = df.groupby(['model_name_short', 'benchmark_name'])[['accuracy']].mean()
    benchmark_accuracy = df.groupby(['agent_name', 'benchmark_name'])[['accuracy']].mean()
    model_accuracy.to_csv('model_accuracy.csv')
    benchmark_accuracy.to_csv('benchmark_accuracy.csv')

if __name__ == "__main__":
    main()
