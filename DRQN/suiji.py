import sys
from typing import Dict, List, Tuple

import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from ENV_UAV import Environ


# from SegmentTree import MinSegmentTree, SumSegmentTree


# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.Linear1 = nn.Linear(state_space, 256)
        # self.lstm    = nn.LSTMCell(256,256)
        self.Linear2 = nn.Linear(256, 256)
        self.Linear3 = nn.Linear(256, action_space)
        self.N_action = action_space

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        return self.Linear3(x)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.N_action - 1)
        else:
            return self.forward(obs).argmax().item()


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def put(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


def train(q_net=None, target_q_net=None, replay_buffer=None,
          device=None,
          optimizer=None,
          batch_size=64,
          learning_rate=1e-3,
          gamma=0.99):
    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples = replay_buffer.sample()

    states = torch.FloatTensor(samples["obs"]).to(device)
    actions = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
    rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
    next_states = torch.FloatTensor(samples["next_obs"]).to(device)
    dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

    # Define loss
    q_target_max = target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
    targets = rewards + gamma * q_target_max * dones
    q_out = q_net(states)
    q_a = q_out.gather(1, actions)

    # Multiply Importance Sampling weights to loss
    loss = F.smooth_l1_loss(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_model(model, path='default.pth'):
    torch.save(model.state_dict(), path)


def smooth(data, sm=1):
    smooth_data = []
    # if sm > 1:
    #     for d in data:
    z = np.ones(len(data))
    y = np.ones(sm) * 1.0
    d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
    smooth_data.append(d)
    return smooth_data


def plot(cost_list):
    y_data = smooth(cost_list, 19)
    x_data = np.arange(len(cost_list))
    # sns.set(style="darkgrid", font_scale=1.5)
    # sns.tsplot(time=x_data, data=y_data, color='b', linestyle='-')
    np.savetxt('suiji.txt',y_data[0],fmt='%f')

    plt.plot(x_data, y_data[0])
    plt.ylabel('suiji-reward')
    plt.xlabel('training Episode')
    # plt.ylim(0.5, 1.0)
    plt.show()

    plt.plot(x_data, cost_list)
    plt.ylabel('reward')
    plt.xlabel('training Episode')
    # plt.ylim(0.5, 1.0)
    plt.show()


if __name__ == "__main__":

    # Determine seeds
    model_name = "DQN"
    env_name = "CartPole-v1"
    seed = 1
    exp_num = 'SEED' + '_' + str(seed)

    # Set gym environment
    env = Environ()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # np.random.seed(seed)
    # random.seed(seed)
    # seed_torch(seed)
    # env.seed(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/' + env_name + "_" + model_name + "_" + exp_num)

    # Set parameters
    batch_size = 64
    learning_rate = 2e-3
    buffer_len = int(50000)
    min_buffer_len = batch_size
    episodes = 2500
    print_per_iter = 20
    target_update_period = 4
    eps_start = 1.0
    eps_end = 0.001
    eps_decay = 0.996
    tau = 1 * 1e-2
    max_step = 100

    lmz = []

    # Create Q functions
    Q = []
    Q_target = []
    state_space = 4
    action_space = env.action_dim
    for i in range(env.n_ch):
        Q.append(Q_net(state_space=state_space, action_space=action_space).to(device))
        Q_target.append(Q_net(state_space=state_space, action_space=action_space).to(device))
    # Q = Q_net(state_space=21, action_space=64).to(device)
    # Q_target = Q_net(state_space=21, action_space=64).to(device)
    for i in range(env.n_ch):
        Q_target[i].load_state_dict(Q[i].state_dict())
    #
    # Q_target.load_state_dict(Q.state_dict())
    replay_buffer = []
    for i in range(env.n_ch):
        replay_buffer.append(ReplayBuffer(state_space, size=buffer_len, batch_size=batch_size))

    # Set optimizer
    score = 0
    # score_sum = 0
    lmz = []

    # optimizer = optim.Adam(Q.parameters(), lr=learning_rate)
    optimizer = []
    for i in range(env.n_ch):
        optimizer.append(optim.Adam(Q[i].parameters(), lr=learning_rate))

    epsilon = eps_start

    # episode_memory = EpisodeMemory(random_update=random_update,
    #                                max_epi_num=100, max_epi_len=600,
    #                                batch_size=batch_size,
    #                                lookup_step=lookup_step)
    # episode_memory = []
    # for i in range(env.n_ch):
    #     episode_memory.append(EpisodeMemory(random_update=random_update,
    #                                         max_epi_num=100, max_epi_len=600,
    #                                         batch_size=batch_size,
    #                                         lookup_step=lookup_step))

    # Train
    for i in range(episodes):
        s = env.reset()
        done = False

        episode_record = []
        a = []
        obs = [[], [], []]
        obs_prime = [[], [], []]

        for t in range(max_step):
            # if i % print_per_iter == 0:
            #     env.render()

            # Get action
            # s = env.get_state()
            for j in range(env.n_ch):
                # obs[j] = np.hstack((s[j][(8 + 2 * j):(10 + 2 * j)],
                                    # s[j][(14 + 2 * j):(16 + 2 * j)]))
                # obs[j] = s[j][8:]
                # obs[j] = np.hstack((s[j][(8 + 2 * j):(10 + 2 * j)],
                #                     s[j][(14 + 2 * j):(16 + 2 * j)]))
                # a.append(Q[j].sample_action(torch.from_numpy(obs[j]).float().to(device), epsilon))
                # obs[j] = np.hstack((s[j][(env.n_channel * env.n_des + env.n_des * j):(
                #         env.n_channel * env.n_des + env.n_des + env.n_des * j)],
                #                           s[j][
                #                           (env.n_channel * env.n_des + env.n_ch * env.n_des + env.n_des * j):(
                #                                   env.n_channel * env.n_des + env.n_ch * env.n_des + env.n_des + env.n_des * j)]))
                a.append(random.randint(0, env.action_dim - 1))
            # a = Q.sample_action(torch.from_numpy(s).float().to(device), epsilon)

            # Do action
            s_prime, r, done, _ = env.step(a)
            # r += s_prime[0] ## For MountainCar

            # make data
            done_mask = 0.0 if done else 1.0

            # for j in range(env.n_ch):
            #     # obs_prime[j] = s_prime[j]
            #     # obs_prime[j] = np.hstack((s_prime[j][(8 + 2 * j):(10 + 2 * j)],
            #     #                     s_prime[j][(14 + 2 * j):(16 + 2 * j)]))
            #     # obs_prime[j] = s[j][8:]
            #     obs_prime[j] = np.hstack((s_prime[j][(8 + 2 * j):(10 + 2 * j)],
            #                         s_prime[j][(14 + 2 * j):(16 + 2 * j)]))
            #     replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, obs_prime[j], done_mask)
            # replay_buffer.put(s, a, r/100.0, s_prime, done_mask)

            s = s_prime

            score += r.sum() / env.n_ch
            # score_sum += r

            # if len(replay_buffer[0]) >= min_buffer_len:
            #     for j in range(env.n_ch):
            #         train(Q[j], Q_target[j], replay_buffer[j], device,
            #               optimizer=optimizer[j],
            #               batch_size=batch_size,
            #               learning_rate=learning_rate)
            #     # train(Q, Q_target, replay_buffer, device,
            #     #         optimizer=optimizer,
            #     #         batch_size=batch_size,
            #     #         learning_rate=learning_rate)
            #
            #     if (t + 1) % target_update_period == 0:
            #         # Q_target.load_state_dict(Q.state_dict()) <- naive update
            #         for j in range(env.n_ch):
            #             for target_param, local_param in zip(Q_target[j].parameters(),
            #                                                  Q[j].parameters()):  # <- soft update
            #                 target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

            if done:
                break

        # epsilon = max(eps_end, eps_decay ** i)  # Linear annealing

        # if i % print_per_iter == 0 and i!=0:
        #     print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
        #                                                     i, score_sum/print_per_iter, len(replay_buffer), epsilon*100))
        # score_sum=0.0
        # save_model(Q, model_name+"_"+exp_num+'.pth')

        # Log the reward
        # writer.add_scalar('Rewards per episodes', score, i)
        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            i, score, len(replay_buffer[0]), epsilon * 100))
        lmz.append(score)
        score = 0

    plot(lmz)
    # writer.close()
    # env.close()