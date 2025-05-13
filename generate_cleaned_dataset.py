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
all_data.to_csv('cleaned_all_metrics.csv', index=False)