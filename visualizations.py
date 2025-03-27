import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

benchmark_win_rates = pd.read_csv('benchmark_win_rates.csv')
model_win_rates = pd.read_csv('model_win_rates.csv')
grouped_df = benchmark_win_rates.groupby('benchmark_name')
dfs_dict = {category: group for category, group in grouped_df}
sns.set_theme(style="whitegrid", palette="muted") 

def benchmark_win_rate_bar(dfs_dict):
    """
    Plot the win rates for each model by benchmark.
    
    Args:
        agg_df: Dictionary of dataframes with win rates for each model
    """
    for category, dataframe in dfs_dict.items():
        # Sort by win rate
        sorted_df = dataframe.sort_values('overall_win_rate', ascending=False)
        # Create bar plot
        ax = sns.barplot(x='model_name_short', y='overall_win_rate', data=sorted_df, hue='model_name_short', palette='magma')
        plt.title(f'Model win rates for {category}')

        plt.xlabel('Model')
        plt.ylabel('Win Rate')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Add win rate values to bar plot
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

        # Add horizontal gridlines
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        category = category.split('/')[-1] 
        plt.savefig(f'visualizations/plots_by_benchmark/{category}_model_win_rates')
        plt.close()

def benchmark_win_rate_bar_full(grouped_df):
    """
    Plot the win rates for each model by benchmark as a grid.
    
    Args:
        grouped_df: DataFrame grouped by benchmark
    """
    sorted_groups = [group.sort_values(by='overall_win_rate', ascending=False) for _, group in grouped_df]

    num_groups = len(sorted_groups)
    # Number of columns in the grid
    cols = 5  
    rows = (num_groups // cols) + (num_groups % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

    # Flatten the axes array
    axes = axes.flatten()

    for i, (group) in enumerate(sorted_groups):
        category = group['benchmark_name'].iloc[0]
        ax = axes[i]
        ax = sns.barplot(x='model_name_short', y='overall_win_rate', data=group, hue='model_name_short', palette='magma', ax=ax) # Create a bar plot for each group
        ax.set_title(f'{category}', fontsize=14)
        ax.set_xlabel('Model')
        ax.set_ylabel('Win Rate')
        ax.tick_params(axis='x', rotation=90, labelsize=7)

        # Add win rate values to bar plot
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=5, color='black', xytext=(0, 5), textcoords='offset points')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    # Fix spacing
    plt.tight_layout()

    plt.savefig('visualizations/grouped_bar_plots.png', dpi=300)


def model_win_rate_bar(model_win_rates):
    """
    Plot the win rates for each model.
    
    Args:
        model_win_rates: DataFrame with win rates for each model
    """

    plt.figure(figsize=(12, 6))
    # Sort by win rate
    sorted_df = model_win_rates.sort_values('overall_win_rate', ascending=False)
    # Create bar plot
    ax = sns.barplot(x='model_name_short', y='overall_win_rate', data=sorted_df, hue='model_name_short', palette='magma')
    
    plt.title('Model Win Rates')
    plt.xlabel('Model')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Add win rate values to bar
    for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

    # Add horizontal gridlines
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f'visualizations/model_win_rates.png')
    

def create_tables(grouped_df):
    """
    Create a table with win rates for each model by benchmark
    
    Args:
    
        grouped_df: DataFrame grouped by benchmark
    """
    grouped_output_file = 'visualizations/tables_by_benchmark.md'

    # Open the file 
    with open(grouped_output_file, 'w') as file:
        # For each group and save the table to the file
        for category, group in grouped_df:
            group = group.drop(columns=['benchmark_name', 'win_rate_mean', 'win_rate_std', 'win_rate_count', 'wins_sum', 'comparisons_sum'])
            file.write(f"\n{category} win rate:\n")
            # Format the table and write it to the file
            formatted_table = tabulate(group, headers='keys', tablefmt='pretty', showindex=False)
            file.write(formatted_table + "\n\n")

def grid_scatter_by_benchmark(tasks, merged_df, x_axis, y_axis, x_label, y_label, num_cols, filename):
    """
    Create scatter plots in a grid comparing metrics for each benchmark
    
    Args:
        tasks: names of benchmarks to group by
        merged_df: dataframe with data to plot
        x_axis: metric 1 being compared
        y_axis: metric 2 being compared
        x_label: x-axis label
        y_label: y-axis label
        num_cols: number of columns in grid
        filename: final plot name
    """

    # Calculate the grid dimensions based on the number of tasks
    n_tasks = len(tasks)
    n_cols = min(num_cols, n_tasks)  # Maximum 3 columns
    n_rows = (n_tasks + n_cols - 1) // n_cols  
    
    # Create a single figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(35, 5 * n_rows))
    
    if n_tasks == 1:
        axes = np.array([axes])
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        df_t = merged_df[merged_df['benchmark_name'] == task]
        benchmark_name = task 
    
        # Create scatter plot
        scatter = ax.scatter(df_t[x_axis], df_t[y_axis], alpha=0.5)
    
        # Annotate each point with the model name
        for i, row in df_t.iterrows():
            ax.annotate(row['model_name_short'], 
                        (row[x_axis], row[y_axis]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8)
    
        ax.set_title(benchmark_name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    # Hide any unused subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()  # Adjust layout to make room for annotations
    plt.savefig('visualizations/tokens_win_rate.png', dpi=300)




# model_win_rate_bar(model_win_rates)
# benchmark_win_rate_bar_full(grouped_df)
# benchmark_win_rate_bar(dfs_dict)
# create_tables(grouped_df)
tokens_win_rate()


