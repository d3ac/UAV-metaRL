from ENV_UAV import Environ
import math
import numpy as np
import random
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_decay(epi_iter):
        decay = math.pow(0.998, epi_iter)  # math.pow(a,b) a的b次方
        if decay < 0.05:
            decay = 0.05
        return decay

################### SETTINGS ######################
IS_TEST = 1
label = 'marl_model'#多智能体强化学习
setup_seed(2022)
n_episode = 2500
n_steps = 200
env = Environ()
train_curve = []
total_train_steps = 0
n_episode_test = 100  # test episodes
cost_list = []   #绘制收敛曲线的列表
steps = 0        #总步数

# ------------------------ initialize ----------------------------
num_input = 4
num_output = env.action_dim
n_agent = env.n_ch

ep_rewards = []
if IS_TEST:
    steps = 0
    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        obs = env.reset()
        hidden_all = []
        for i in range(n_agent):
            hidden = (torch.zeros(1, 1, 16).float(), torch.zeros(1, 1, 16).float())
            hidden_all.append(hidden)
        terminal = False
        episode_reward = 0
        reward = 0

        for step in range(n_steps):
            state_old_all = []
            action_all = [0,0,0]
            state_old_all = env.get_state() # 3 * 23(8个sci, 6个channel, 6个power, 3个干扰机)
            obs = [[], [], []]
            for i in range(n_agent):
                obs[i] = np.hstack(( # 横着拼起来
                    # 4 channel, 2 n_des, 
                    state_old_all[i][(env.n_channel * env.n_des + env.n_des * i):(env.n_channel * env.n_des + env.n_des * i + env.n_des)], # 和簇成员 uav channel 2个 & uav power 2个
                    state_old_all[i][(env.n_channel * env.n_des + env.n_ch * env.n_des + env.n_des * i):(env.n_channel * env.n_des + env.n_ch * env.n_des + env.n_des + env.n_des * i)]
                ))
                action_temp, hidden = env.agents[i].get_action(obs[i], hidden_all[i], get_decay(i_episode))
                action_all[i] = action_temp
                hidden_all[i] = hidden

            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all.copy()
            obs_, train_reward, terminal, info = env.step(action_temp)
            for i in range(n_agent):
                env.agents[i].remember(obs[i], action_temp[i], train_reward.sum() / n_agent)
            # obs = obs_
            episode_reward += (train_reward.sum(axis=0) / n_agent)
            total_train_steps += 1
            steps += 1
            # print("steps:", steps)
            if env.agents[0].buffer.is_available() and i_episode > 50:    #随便检查一个的经验池即可
                for i in range(n_agent):
                    env.agents[i].train()
        for i in range(n_agent):
            env.agents[i].buffer.create_new_epi()

        ep_rewards.append(episode_reward)
        print("reward:", episode_reward)
        # print('ep_rewards', ep_rewards)

    env.plot(ep_rewards)