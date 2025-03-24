from abc import ABC
import json 
from glob import glob 
import os 
import pandas as pd 
import numpy as np
from .config import DATA_DIR

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

        self.config['agent_name_short'] = self.config['agent_name'].split(' (')[-2]
        self.config['model_name_short'] = self.config['agent_name'].split(' (')[-1].split(')')[0]
    
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
        return {'accuracy': self.return_accuracy()}

    def return_accuracy(self):
        return self.data['results']['accuracy']

    def pass_at_k(self, n, k):
        """
        :param n: total number of samples
        :param c: number of correct samples
        :param k: k in pass@$k$
        """
        # calculate c here # 

        #                  
        pass 
        # if n - c < k: return 1.0
        # return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def return_

    def pass_to_the_k():
        pass 

    def win_rate(self):
        pass

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
