import subprocess
import os

#### SCRIPT DESCRIPTIONS ####
# 1. generate_cleaned_dataset.py
    # Generates a CSV file dataset from the traces into the cluster, with agent name, benchmark name, model name, accuracy, cost, and latency
# 2. check_data.py
    # Checks that all the runs in our status spreadsheet are successfully in the dataset
# 3. cost_accuracy_curve.py
    # Creates the pareto frontiers by benchmark of cost vs. accuracy
    # Saves the distance of each point from the pareto frontier per benchmark in visualizations/pareto_distances/pareto_distances.csv
# 4. win_rates.py – this outputs two files:
    # model_win_rates_max.csv: overall model win rates across benchmarks calculated using max accuracy
    # model_win_rates_max_pareto.csv: overall model win rates across benchmarks calculated using distance from the pareto frontier
# 5. convex_hull.py – this will create the pareto frontiers for:
    # latency vs. accuracy per benchmark
    # cost vs. win rate calculated using max accuracy
    # cost vs. win rate calculated using distance from the pareto frontier
    # latency vs. win rate calculated using max accuracy 
    # latency vs. win rate calculated using distance from the pareto frontier
# 6. visualizations.py
    # A bar plot for win rates calculated using max accuracy for models across benchmarks
    # A bar plot for win rates calculated using using distance from the pareto frontier for models across benchmarks
    # Six heatmaps: (generalist agent scaffold, task-specific agent scaffold) x (latency, cost, accuracy)
# NOTES:
    # All plots can be found in the visualizations/new_plots folder.
    # If you already have the cleaned dataset ready, you can remove/comment out generate_cleaned_dataset.py and check_data.py from 


pipeline_scripts = ["generate_cleaned_dataset.py", "check_data.py", "cost_accuracy_curve.py", "win_rates.py", "convex_hull.py", "visualizations.py"]

python_executable = "python"

def run_script(script_path):
    print(f"Running: {script_path}")
    try:
        result = subprocess.run(
            [python_executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"{script_path} ran successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"{script_path} failed with exit code {e.returncode}.")
        print(f"Error output:\n{e.stderr}")
        return False, e.stderr

def main():
    for script in pipeline_scripts:
        if not os.path.exists(script):
            print(f"Script not found: {script}")
        success, output = run_script(script)

if __name__ == "__main__":
    main()
