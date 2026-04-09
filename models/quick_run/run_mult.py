import json
import os
import sys
import MMSA
import torch
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    #由于这里经常跑一半跑失败我做了一点限制
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(4, cpu_count // 2)  
    print(f"CPU cores: {cpu_count}, Setting num_workers to: {optimal_workers}")

    CONFIG = json.loads("./Configs/tfn.json")
    MMSA.run(
        model_name='mult',
        dataset_name='sims',
        gpu_ids=[0],
        config = CONFIG
    )