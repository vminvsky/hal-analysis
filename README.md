<h3 align="center">HAL Analysis</h3>

  <p align="center">
    Analysis of HAL traces
    <br />
    <br />
    <br />
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Analysis of agent traces



<!-- GETTING STARTED -->
## Getting Started

### Setup

1. Install the HAL harness:
```
git clone --recursive https://github.com/benediktstroebl/hal-harness.git
cd hal-harness
```
2. Create conda environment:
```
conda create -n hal python=3.12
conda activate hal
```
3. Install the `hal` package:
```
pip install -e .
cd ..
```
4. Download and decrypt the traces: 
```
huggingface-cli download agent-evals/hal_traces --repo-type dataset
hal-decrypt -D path/to/directory
```
where `path/to/directory` is the path to where the data files were downloaded.

4. Clone this repository:
```
 git clone https://github.com/vminvsky/hal-analysis.git
 cd hal-analysis
 pip install -r requirements.txt
 ```

 5. Set `DATA_DIR` in `config.py` to `'path/to/directory'`.
 ```
 DATA_DIR = 'path/to/directory'
 ```
 where `path/to/directory` is the path to where the data files were downloaded.

 6. Set `files` in `generate_cleaned_dataset.py` to:
 ```
 files = glob('path/to/directory/*.json')
 ```

<!-- USAGE -->
## Usage
Run `run_pipeline.py`. This will run the following scripts:
1. `generate_cleaned_dataset.py`
2. `check_data.py`
3. `cost_accuracy_curve.py`
4. `win_rates.py`
5. `convex_hull.py`
6. `visualizations.py`

You can update the agent/model name mappings in `config.py`. 

## Visualizations generated
1. `visualizations/new_plots/heatmaps/dist_overall_win_rate_heatmap.png` - heatmap of win rates calculated using the distance from pareto frontier of the Cost vs. Max Accuracy over all agent scaffolds
2. `visualizations/new_plots/heatmaps/generalist_accuracy_heatmap.png`
3. `visualizations/new_plots/heatmaps/generalist_mean_latency_heatmap.png` - heatmap of the latencies for each model/benchmark pair for generalist agents (latency per benchmark calculated by taking the mean of the latency of each task)
4. `visualizations/new_plots/heatmaps/generalist_overall_win_rate_heatmap.png` - heatmap of win rates of models with the generalist agent scaffold; win rate for each benchmark is calculated as the proportion of times a model has a higher accuracy than other models with generalist scaffolds within the benchmark 
5. `visualizations/new_plots/heatmaps/generalist_total_cost_heatmap.png`
6. `visualizations/new_plots/heatmaps/max_acc_overall_win_rate_heatmap.png` - heatmap of win rates calculated as: using the max accuracy of the model across all agent scaffolds for a particular benchmark, the proportion of times a model has a higher accuracy than other models with the highest accuracy scaffold within the benchmark 
7. `visualizations/new_plots/heatmaps/task_specific_accuracy_heatmap.png`
8. `visualizations/new_plots/heatmaps/task_specific_mean_latency_heatmap.png` - heatmap of the latencies for each model/benchmark pair for task specific agents (latency per benchmark calculated by taking the mean of the latency of each task)
9. `visualizations/new_plots/heatmaps/task_specific_overall_win_rate_heatmap.png` - heatmap of win rates of models with the task specific agent scaffold; win rate for each benchmark is calculated as the proportion of times a model has a higher accuracy than other models with task specific scaffolds within the benchmark 
10. `visualizations/new_plots/heatmaps/task_specific_total_cost_heatmap.png`
11. `visualizations/new_plots/convex_model_cost_accuracy.png` - accuracy vs. total cost paretos for each bechmark, using max accuracy across agent scaffolds for each model 
12. `visualizations/new_plots/convex_model_latency_accuracy.png` - accuracy vs. latency paretos for each bechmark, using max accuracy across agent scaffolds for each model 
13. `visualizations/new_plots/cost_win_rate_max.png` - pareto frontier of win rate across all benchmarks calculated using the max accuracy across agents vs mean cost across all benchmarks 
14. `visualizations/new_plots/cost_win_rate_pareto.png` - pareto frontier of win rate across all benchmarks vs mean cost across all benchmarks; win rate across benchmarks is calculated as the proportion of times a model has a higher accuracy than other models with the highest accuracy scaffold across benchmarks
15. `visualizations/new_plots/latency_win_rate_max.png` -  pareto frontier of win rate across all benchmarks calculated using the max accuracy across agents vs mean latency of all tasks across all benchmarks 
16. `visualizations/new_plots/latency_win_rate_pareto.png` - pareto frontier of win rate across all benchmarks calculated  using the distance from pareto frontier of the cost vs. max accuracy over all agent scaffolds vs. mean latency of all tasks across all benchmarks 
17. `visualizations/new_plots/model_win_rates_max.png` - win rates of each model calculated using the max accuracy across agents
18.  `visualizations/new_plots/model_win_rates_pareto.png` - win rates of each model calculated using the distance from pareto frontier of the cost vs. max accuracy over all agent scaffolds 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Contributors:

<a href="https://github.com/vminvsky/hal-analysis/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vminvsky/hal-analysis" alt="contrib.rocks image" />
</a>
