import pandas as pd

# Read the CSV files
agent_run_status = pd.read_csv('data/agent_run_status.csv')
cleaned_dataset = pd.read_csv('data/cleaned_all_metrics.csv')

agent_run_status = agent_run_status[agent_run_status['Status'] == 'Uploaded']

model_map = {
    'claude-3-7-sonnet-20250219': 'claude-3-7-sonnet-2025-02-19',
    'claude-3-7-sonnet-20250219-thinking-high': 'claude-3-7-sonnet-2025-02-19 high',
    'claude-3-7-sonnet-20250219-thinking-low': 'claude-3-7-sonnet-2025-02-19 low',
    'claude-3-7-sonnet-20250219-thinking-low': 'claude-3-7-sonnet-2025-02-19 low',
    'o4-mini-2025-04-16-medium': 'o4-mini-2025-04-16'
}

benchmark_map = {
    'swebench-verified-mini': 'swebench_verified_mini',
    'colbench-backend': 'colbench_backend_programming',
    'colbench-frontend': 'colbench_frontend_design',
    'ScienceAgentBench': 'scienceagentbench',
    'Online-Mind2Web': 'online_mind2web'
}

scaffold_map = {
    'TAU-bench FewShot Agent': 'TAU-bench FewShot',
    'Open Deep Research': 'HF Open Deep Research',
    'Reflexion': 'USACO Episodic + Semantic',
    'Colbench Agent': 'Col-bench Text',
    'SciCode Tool Calling': 'Scicode Tool Calling Agent',
    'Scicode Zero Shot': 'Scicode Zero Shot Agent',
    'Assistantbench Browsing Agent': 'Assistantbench Browser Agent'
}

agent_run_status['Model'] = agent_run_status['Model'].replace(model_map)
agent_run_status['Benchmark'] = agent_run_status['Benchmark'].replace(benchmark_map)
agent_run_status['Agent'] = agent_run_status['Agent'].replace(scaffold_map)

agent_run_status['Model'] = agent_run_status['Model'].str.replace('-low', ' low', regex=False)
agent_run_status['Model'] = agent_run_status['Model'].str.replace('-high', ' high', regex=False)
agent_run_status['Model'] = agent_run_status['Model'].str.replace('-medium', ' medium', regex=False)

models_to_remove = ['2.5-pro', 'o1', 'o3-mini', 'gpt-4o', 'o3-2025-04-16 low', 'claude-3-7-sonnet-2025-02-19 low']
pattern = '|'.join(models_to_remove)
agent_run_status = agent_run_status[~agent_run_status['Model'].str.contains(pattern, case=False, na=False)]
agent_run_status = agent_run_status[~(agent_run_status['Model'] == 'o4-mini-2025-04-16')]
# Select and rename columns to match
agent_run_status = agent_run_status[['Benchmark', 'Agent', 'Model']]
cleaned_dataset = cleaned_dataset[['agent_name_short','model_name_short', 'benchmark_name']]

agent_run_status = agent_run_status.rename(columns={
    'Benchmark': 'benchmark_name', 
    'Agent': 'agent_name_short', 
    'Model': 'model_name_short'
})

# Define key columns (for grouping) and value column (for comparison)
key_columns = ['agent_name_short', 'benchmark_name']
value_column = 'model_name_short'

# Create composite keys for comparison (handles multiple rows with same grouping values)
agent_run_status['composite_key'] = agent_run_status[key_columns + [value_column]].apply(
    lambda x: tuple(x), axis=1
)
cleaned_dataset['composite_key'] = cleaned_dataset[key_columns + [value_column]].apply(
    lambda x: tuple(x), axis=1
)

# Find rows only in agent_run_status
only_in_agent = agent_run_status[~agent_run_status['composite_key'].isin(cleaned_dataset['composite_key'])]
only_in_agent = only_in_agent.drop('composite_key', axis=1)

# Find rows only in cleaned_dataset
only_in_cleaned = cleaned_dataset[~cleaned_dataset['composite_key'].isin(agent_run_status['composite_key'])]
only_in_cleaned = only_in_cleaned.drop('composite_key', axis=1)

print(f"MISSING ROWS: Rows in agent_run_status but missing from cleaned_dataset ({len(only_in_agent)} rows):")
if len(only_in_agent) > 0:
    print(only_in_agent.to_string(index=False))
else:
    print("No missing rows found.")

print(f"\nEXTRA ROWS: Rows in cleaned_dataset that shouldn't be there ({len(only_in_cleaned)} rows):")
if len(only_in_cleaned) > 0:
    print(only_in_cleaned.to_string(index=False))
else:
    print("No extra rows found.")

output_lines = []
output_lines.append("MISSING ROWS ANALYSIS")
output_lines.append(f"Rows in agent_run_status but missing from cleaned_dataset ({len(only_in_agent)} rows):\n")

if len(only_in_agent) > 0:
    output_lines.append(only_in_agent.to_string(index=False))
else:
    output_lines.append("No missing rows found.")

output_lines.append(f"\n\nEXTRA ROWS: Rows in cleaned_dataset not in agent run status ({len(only_in_cleaned)} rows):\n")
if len(only_in_cleaned) > 0:
    output_lines.append(only_in_cleaned.to_string(index=False))
else:
    output_lines.append("No extra rows found.")

# Add summary to output
output_lines.append("\n\n" + "="*50)
output_lines.append("SUMMARY")
output_lines.append("="*50)
output_lines.append(f"Total rows in agent_run_status: {len(agent_run_status)}")
output_lines.append(f"Total rows in cleaned_dataset: {len(cleaned_dataset)}")
output_lines.append(f"Rows missing from cleaned_dataset: {len(only_in_agent)}")
output_lines.append(f"Extra rows in cleaned_dataset: {len(only_in_cleaned)}")

# Write to file
with open('data/missing_rows_analysis.txt', 'w') as f:
    f.write('\n'.join(output_lines))

print("\n")
print("SUMMARY")
print(f"Total rows in agent_run_status: {len(agent_run_status)}")
print(f"Total rows in cleaned_dataset: {len(cleaned_dataset)}")
print(f"Rows missing from cleaned_dataset: {len(only_in_agent)}")
print(f"Extra rows in cleaned_dataset: {len(only_in_cleaned)}")
print(f"\nResults saved to: missing_rows_analysis.txt")