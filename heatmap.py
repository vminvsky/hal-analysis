import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# To-do:
### make this a function so we can run for all 6 heatmaps we want

df = pd.read_csv('cleaned_all_metrics.csv')
cols = ['benchmark_name', 'agent_name_short', 'model_name_short', 'total_cost']
df = df[cols].copy()
# add code to only keep rows if they are generalist agents, vs. only keep if they are task specific agents depending on what we want
df = df.pivot(columns = 'benchmark_name', index = 'model_name_short', values = 'total_cost')

df['benchmarks_mean'] = df.mean(axis = 1)
df.loc['models_mean'] = df.mean(axis = 0)

fig, ax = plt.subplots()

sns.heatmap(ax = ax, data = df, annot = True, fmt = '.0f')

plt.show()
plt.savefig('visualizations/new_plots/cost_heatmap.png')
