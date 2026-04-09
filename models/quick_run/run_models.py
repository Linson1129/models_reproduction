import json
import os
import sys
import MMSA
import torch
import multiprocessing

if __name__ == '__main__':

    CONFIG = json.loads("./Configs/tfn.json")
    MMSA.run(
        model_name='tfn',
        dataset_name='sims',
        gpu_ids=[0],
        config = CONFIG
    )
