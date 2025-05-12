from abc import ABC
import json 
from glob import glob 
import os 
import pandas as pd 
import numpy as np
from .config import DATA_DIR, MODEL_NAME_MAP, AGENT_NAME_MAP

# final dataset 
# columns: 
# - config details
    # - agent name
    # - dataset name
    # - task name
    # - model name
    # - ...
# - metrics 
    # - accuracy 
    # - pass@k
    # - ...

class DataCombiner:
    def __init__(self, task_name: str, data_dir: str = DATA_DIR):
        self.task_name = task_name
        self.data_dir = data_dir
        self.data_paths = glob(os.path.join(data_dir, f"{task_name}*.json"))

    def load(self):
        self.data = [DataLoader(data_path) for data_path in self.data_paths]
        self.df = pd.DataFrame([d.return_row() for d in self.data])
        return self.df

class DataLoader(ABC):
    """
    abstract method to harmonize all the different data formats 
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_data()
        self.config = self._load_config()

        agent_name = self.config['agent_name']
        # print(agent_name)
        self.config['agent_name'] = AGENT_NAME_MAP.get(agent_name, agent_name)
        self.config['agent_name_short'] = self.config['agent_name'].split(' (')[-2]

        model_name_short = (self.config['agent_name'].split(' (')[-1].split(')')[0])
        self.config['model_name_short'] = MODEL_NAME_MAP.get(model_name_short, model_name_short)
        # print(self.config['agent_name'], " | ", self.config['agent_name_short'], " | ", self.config['model_name_short'])


    def _load_data(self):
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _load_config(self):
        return self.data['config']
    
    def return_row(self):
        return {**self.config, **self.return_metrics()}

    def return_metrics(self):
        # return all the metrics we care about.
        # return {'accuracy': self.return_accuracy(), 'total_tokens': self.return_total_tokens(), 'total_cost': self.return_total_cost(), 'latencies_per_task': self.return_latency()}
        return {'accuracy': self.return_accuracy(), 'total_cost': self.return_total_cost(), 'latencies_per_task': self.return_latency()}

    def return_accuracy(self):
        if ('average_correctness' in self.data['results']):
            return self.data['results']['average_correctness']
        elif ('success_rate' in self.data['results']):
            return self.data['results']['success_rate']
        else:
            return self.data['results']['accuracy']

    def return_total_tokens(self):
        try:
            return next(iter(self.return_token_usages().values()))['total_tokens']
        except:
            try: 
                vals  = next(iter(self.return_token_usages.values()))
                total_tokens = sum(vals.values())
            except:
                pass
            return None
    
    def return_token_usages(self):
        return self.data['total_usage']
    
    def return_total_cost(self):
        return self.data['results']['total_cost']

    def pass_to_the_k():
        pass 

    def return_latency(self):
        latencies = []
        if 'latencies' in self.data['results']:
            for key in self.data['results']['latencies'].keys():
                latencies.append(self.data['results']['latencies'][key]['total_time'])
        return latencies


    def __repr__(self):
        return f"DataLoader(agent_name={self.config['agent_name']}, benchmark_name={self.config['benchmark_name']})"

if __name__ == "__main__": 
    data_path = '/scratch/gpfs/vv7118/models/hub/datasets--agent-evals--agent_traces/snapshots/8831400af880b37c06a15026f661f726160a44c2/taubench_retail_1740413575.json'
    data_loader = DataLoader(data_path)
    print(data_loader.data.keys())
    print(data_loader.return_accuracy())
    print(data_loader.return_row())

    combiner = DataCombiner('taubench_retail')
    df = combiner.load()
    print(df)
