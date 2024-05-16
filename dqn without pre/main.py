import torch
import random
import numpy as np
import random
import tqdm
import math
import pandas as pd
import pickle

from uavenv import Environ

def get_decay(epi_iter):
    decay = math.pow(0.998, epi_iter)
    if decay < 0.05:
        decay = 0.05
    return decay

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def generate_task(n_task, env):
    tasks = []
    for i in range(n_task):
        task = env.generate_p_trans()
        tasks.append(task)
    return tasks

def get_obs(env, obs):
    state = []
    for i in range(env.n_ch):
        state.append(
            np.hstack((
                obs[i][0:10],
                obs[i][(env.n_channel * env.n_des + env.n_des * i):(env.n_channel * env.n_des + env.n_des * i + env.n_des)],
                obs[i][(env.n_channel * env.n_des + env.n_ch * env.n_des + env.n_des * i):(env.n_channel * env.n_des + env.n_ch * env.n_des + env.n_des + env.n_des * i)]
            ))
        )
    return state

if __name__ == '__main__':
    # set_seed(1)
    # ----------------- Training Parameters ----------------- #
    n_episode = 2500
    n_steps = 200
    n_pre_sample = 200 # 先sample一些东西放在memory里
    sample_length = 50

    env = Environ()
    with open('tasks.pkl', 'rb') as f:
        tasks = pickle.load(f)
    train_task = tasks[:80]
    test_task = tasks[80:]

    train_task_list = list(range(len(train_task)))
    test_task_list = list(range(len(test_task)))

    test_reward_list = []
    train_reward_list = []

    Trange = tqdm.tqdm(range(n_episode))
    for i_episode in Trange:
        # ---------------------------------------- train ----------------------------------------
        task_idx = np.random.choice(train_task_list)
        task = train_task[task_idx]
        for i in range(env.n_ch):
            env.agents[i].buffer.reset()
        
        # ----------------- train -----------------        
        train_reward = 0
        next_obs = env.reset(task)
        for step in range(n_steps):
            obs = get_obs(env, next_obs)
            actions = [[], [], []]
            for i in range(env.n_ch):
                actions[i] = env.agents[i].get_action(obs[i], get_decay(i_episode))
            next_obs, rewards, dones, info = env.step(actions)
            for i in range(env.n_ch):
                env.agents[i].remember(obs[i], actions[i], rewards[i], next_obs[i])
                
            train_reward += rewards.sum(axis=0) / env.n_ch
            if env.agents[0].buffer.is_available():
                for i in range(env.n_ch):
                    env.agents[i].train()
        train_reward_list.append(train_reward)

        # ---------------------------------------- test ----------------------------------------
        params = env.get_params()

        task_idx = np.random.choice(test_task_list)
        task = test_task[task_idx]
        for i in range(env.n_ch):
            env.agents[i].buffer.reset()
        
        # ----------------- train -----------------
        test_reward = 0
        next_obs = env.reset(task)
        for step in range(n_steps):
            obs = get_obs(env, next_obs)
            actions = [[], [], []]
            for i in range(env.n_ch):
                actions[i] = env.agents[i].get_action(obs[i], get_decay(i_episode))
            next_obs, rewards, dones, info = env.step(actions)
            for i in range(env.n_ch):
                env.agents[i].remember(obs[i], actions[i], rewards[i], next_obs[i])
            test_reward += rewards.sum(axis=0) / env.n_ch
            if env.agents[0].buffer.is_available():
                for i in range(env.n_ch):
                    env.agents[i].train()
        test_reward_list.append(test_reward)
        Trange.set_postfix(train_reward=train_reward, test_reward=test_reward)

        env.load_params(params)
        DataFrame = pd.DataFrame([train_reward_list, test_reward_list], index = ['train', 'test']).T
        DataFrame.to_csv('reward.csv', index=False)