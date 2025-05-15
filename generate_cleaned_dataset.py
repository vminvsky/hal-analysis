import pandas as pd
import json 
from glob import glob 

from src.dataloaders.abstract import DataLoader


files = glob('/scratch/gpfs/vv7118/models/hub/datasets--agent-evals--hal_traces/snapshots/597b22d25a9649ff7f5c97f7115fb8a158961709/*.json')

all_data = []
for file in files:
    data = DataLoader(file)
    all_data.append(data.return_row())

all_data = pd.DataFrame(all_data)
models_to_remove = ['2.5-pro', 'o1', 'o3-mini', 'gpt-4o']
pattern = '|'.join(models_to_remove)
all_data_filtered = all_data[~all_data['model_name_short'].str.contains(pattern, case=False, na=False)]

all_data_filtered.to_csv('cleaned_all_metrics.csv', index=False)