
from pareto_utils import (
    grid_pareto_frontier_by_benchmark,
    save_pareto_distances,
    get_max_accuracy )

def cost_accuracy():
    df_m = get_max_accuracy()
    tasks = df_m['benchmark_name'].unique()
    
    # Generate plots for cost vs accuracy
    grid_pareto_frontier_by_benchmark(tasks, df_m, 'total_cost', 'accuracy', 
        'Total Cost', 'Accuracy', 5, 'model_cost_accuracy.png')
    
    # Save Pareto distances
    save_pareto_distances(df_m, tasks, 'total_cost', 'accuracy')

if __name__ == "__main__":
    cost_accuracy()
