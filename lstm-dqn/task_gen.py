import torch
import random
import numpy as np
import random
import tqdm
import math
import pandas as pd
import pickle

from uavenv import Environ

def generate_task(n_task, env):
    tasks = []
    for i in range(n_task):
        task = env.generate_p_trans()
        tasks.append(task)
    return tasks

if __name__ == '__main__':
    env = Environ()
    tasks = generate_task(100, env)
    with open('tasks.pkl', 'wb') as f:
        pickle.dump(tasks, f)