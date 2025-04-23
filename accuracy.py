import pandas as pd

from src.dataloaders.abstract import DataCombiner

def main():
    # Load data for all tasks you want to analyze
    tasks = ['taubench_airline']
             # , 'usaco', 'test','taubench', 'swebench', 'react', 'planexec', 'ipfuncall', 'inspect', 'gaia', 'fullcode', 'cybench', 'agentharm_', 'agentharm_benign']  # Add your task names here
    # tasks = ['gaia', 'cybench', 'taubench', 'taubench_retail' , 'agentharm_']
    cols = ['model_name_short', 'accuracy', 'benchmark_name', 'agent_name', 'agent_name_short']
    all_data = []
    all_data = pd.DataFrame()
    for task in tasks:
        try:
            combiner = DataCombiner(task)
            df = combiner.load()
            df = df[cols]
            all_data = pd.concat([all_data, df])
        except Exception as e:
            print(f"Error loading task {task}: {e}")

    if all_data.empty:
        print("No data found for any task")
        return

    df = all_data
    print(df.keys())
    model_accuracy = df.groupby(['model_name_short', 'benchmark_name'])[['accuracy']].mean()
    benchmark_accuracy = df.groupby(['agent_name_short', 'benchmark_name'])[['accuracy']].mean()
    model_accuracy.to_csv('model_accuracy.csv')
    benchmark_accuracy.to_csv('benchmark_accuracy.csv')

    # mean accuracy of models across benchmarks
    model_mean_accuracy = model_accuracy.groupby('model_name_short')['accuracy'].mean().reset_index()
    model_mean_accuracy = model_mean_accuracy.rename(columns={'accuracy':'mean_accuracy'})
    model_mean_accuracy.to_csv('data/model_mean_accuracy.csv')

    # mean accuracy of agents across benchmarks
    agent_mean_accuracy = benchmark_accuracy.groupby('agent_name_short')['accuracy'].mean().reset_index()
    agent_mean_accuracy = agent_mean_accuracy.rename(columns={'accuracy':'mean_accuracy'})
    agent_mean_accuracy.to_csv('data/agent_mean_accuracy.csv')

if __name__ == "__main__":
    main()
