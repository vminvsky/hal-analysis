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
1. Run `cost_accuracy_curve.py`. This will create the pareto frontiers by benchmark of cost vs. accuracy. It will also save the distance of each point from the pareto frontier per benchmark in `visualizations/pareto_distances/pareto_distances.csv`.
2. Run `win_rates.py`. This outputs two files:
    1. `model_win_rates_max.csv` overall model win rates across benchmarks calculated using max accuracy
    2. `model_win_rates_max_pareto.csv` overall model win rates across benchmarks calculated using distance from the pareto frontier
4. Run `convex_hull.py`. This will create the pareto frontiers for:
    1. latency vs. accuracy pareto per benchmark
    2. cost vs. win rate calculated using max accuracy pareto
    3. cost vs. win rate calculated using distance from the pareto frontier pareto
    4. latency vs. win rate calculated using max accuracy pareto
    5. latency vs. win rate calculated using distance from the pareto frontier pareto
4. Run `visualizations.py`. This will create:
    1. A bar plot for win rates calculated using max accuracy for models across benchmarks
    2. A bar plot for win rates calculated using using distance from the pareto frontier for models across benchmarks
    3. Six heatmaps: (generalist agent scaffold, task-specific agent scaffold) x (latency, cost, accuracy)
   
All plots can be found in the `visualizations/new_plots` folder.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Contributors:

<a href="https://github.com/vminvsky/hal-analysis/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vminvsky/hal-analysis" alt="contrib.rocks image" />
</a>
