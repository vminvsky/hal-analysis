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


### Prerequisites

* prereq 1
  ```sh
  
  ```

<!-- USAGE EXAMPLES -->
## Usage

* `accuracy.py`: outputs two csv files with
  * the accuracy for each model benchmark pair (`model_accuracy.csv`)
  * the accuracy for each agent scaffold benchmark pair (`benchmark_accuracy.csv`)
* `latencies.py`: outputs four csv files with
  * the latency for each model benchmark pair (`model_latency.csv`)
  * the latency for each agent scaffold benchmark pair (`agent_latency.csv`)
  * the mean latencies of each model across all benchmarks (`data/model_mean_latency.csv`)
  * the mean latencies of each agent scaffold across all benchmarks (`data/agent_mean_latency.csv`)
* `convex_hull.py`:
  * plots the convex hull of the pareto frontier per model per benchmark of cost and accuracy as subplots (`visualizations/new_plots/convex_model_cost_accuracy.png`)
  * calculates the AUCs from each subplot and returns as a csv (`visualizations/auc_data/model_cost_accuracy_auc.csv`)
  * plots the AUCs as a bar plot (`visualizations/auc_visualizations/model_cost_accuracy_auc_viz.png`)
  * measures the distance of each model from the convex hull, saves as a csv (`visualizations/pareto_distances/pareto_distances.csv`)
* `token_usage.py`: outputs four csv files with 
  * the cost for each model benchmark pair (`model_total_usage.csv`)
  * the cost for each agent scaffold benchmark pair (`benchmark_total_usage.csv`)
  * the mean costs of each model across all benchmarks (`data/model_mean_cost.csv`)
  * the mean costs of each agent scaffold across all benchmarks (`data/agent_mean_cost.csv`)
* `win_rates.py`: outputs two csv files with:
  * overall model win rates across benchmarks (`model_win_rates.csv`)
  * model win rates by benchmark (`benchmark_win_rates.csv`)
* `visualizations.py`: script that plots all figures for our analysis
* `src/dataloaders/config.py`: mappings to standardize agent and model names across benchmarks
* `visualizations`: folder with visualizations
  * `visualizations/auc_data/model_cost_accuracy_auc.csv`: csv file with auc for the cost accuracy pareto
  * `visualizations/auc_visualizations/model_cost_accuracy_auc_viz.png`: bar plot of auc for the cost accuracy pareto for each benchmark
  * `visualizations/new_plots`: folder with final visualizations
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Contributors:

<a href="https://github.com/vminvsky/hal-analysis/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vminvsky/hal-analysis" alt="contrib.rocks image" />
</a>
