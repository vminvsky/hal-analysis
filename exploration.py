import pandas as pd
import json 
from glob import glob 
from src.dataloaders.abstract import DataLoader
import os

# def sizeof_fmt(num, suffix="B"):
#     for unit in ["", "K", "M", "G", "T"]:
#         if abs(num) < 1024.0:
#             return f"{num:.2f} {unit}{suffix}"
#         num /= 1024.0
#     return f"{num:.2f} P{suffix}"

files = glob("/Users/nn7887/.cache/huggingface/hub/datasets--agent-evals--hal_traces/snapshots/77246dba01c07019ea179a6e9f3b8763520b1d22/assistantbench_assistantbench_browser_agent_o320250416_1746376643_UPLOAD*.json")

all_data = []
# total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
# print("Total size:", sizeof_fmt(total_size))

for file in files:
    data = DataLoader(file)
    all_data.append(data.return_row())

# latencies = []
# for i in range(len(data.data['results']['latencies'].keys())):
#     latencies.append(data.data['results']['latencies'][f'{i}']['total_time'])

# print(data.data['config'].keys())
# print(data.data['config']['agent_args']['evaluator_webjudge_model_name'])
# print(data.data['config']['agent_args']['model_name'])
# print(data.data['results']['latencies']['30']['total_time'])
# print(data.data['results']['latencies'].keys())
# print(data.data['raw_logging_results'].keys())
# print(data.data['results']['accuracy'])
# print(data.data['raw_eval_results'].keys())
# print(data.data['total_usage'].values())
# print(data.data['total_usage'].keys())
# print(data.data['results']['total_cost'])
# print(next(iter(data.data['total_usage'].values()))['total_tokens'])
# print(data.data['raw_eval_results']['aggregate']['task_goal_completion'])