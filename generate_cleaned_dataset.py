import pandas as pd
import json 
from glob import glob 
import numpy as np
from src.dataloaders.abstract import DataLoader

def ensure_list(x):
    # Handle NaN or None values
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # If it's already a list or tuple, convert to list
    if isinstance(x, (list, tuple)):
        return list(x)

    # If it's a string, try to parse it
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, (list, tuple)):
                return list(val)
            else:
                return [val]
        except Exception as e:
            print(f"Failed to parse: {x!r} - Error: {e}")
            return []

    # Fallback: wrap anything else in a list
    return [x]

# Replace with your data directory
files = glob('/Users/nn7887/.cache/huggingface/hub/datasets--agent-evals--hal_traces/snapshots/77246dba01c07019ea179a6e9f3b8763520b1d22/*.json')

all_data = []
for file in files:
    data = DataLoader(file)
    all_data.append(data.return_row())

all_data = pd.DataFrame(all_data)
models_to_remove = ['2.5-pro', 'o1', 'o3-mini', 'gpt-4o', 'o3-2025-04-16 low', 'claude-3-7-sonnet-2025-02-19 low']
pattern = '|'.join(models_to_remove)
all_data_filtered = all_data[~all_data['model_name_short'].str.contains(pattern, case=False, na=False)]
all_data_filtered = all_data_filtered[~(all_data_filtered['model_name_short'] == 'o4-mini-2025-04-16')]

all_data_filtered = all_data_filtered.copy()
all_data_filtered['latencies_per_task'] = all_data_filtered['latencies_per_task'].apply(ensure_list)
all_data_filtered = all_data_filtered.copy()
all_data_filtered['mean_latency'] = all_data_filtered['latencies_per_task'].apply(lambda x: np.mean(x) if x else np.nan)

all_data_filtered.to_csv('data/cleaned_all_metrics.csv', index=False)