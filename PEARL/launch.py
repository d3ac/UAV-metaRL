import os
import pathlib
import numpy as np
import click
import json
import torch
import random

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from default import default_config
from uavenv import Environ
from discret import DiscretPolicy
# ----------------------------- 导入包 -----------------------------



@click.command()
@click.argument('config', default='./configs/point-robot.json')
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
def main(config, gpu, docker):
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    variant = default_config

    # create multi-task environment and sample tasks
    env = Environ() 
    tasks = range(env.n_task)
    obs_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.nvec[0] * 3 #TODO 注意这个地方其实要乘三
    action_dim = 3 * 8
    reward_dim = 1 # 改成1了

    # instantiate networks
    latent_dim = variant['latent_size']
    # context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[64, 64, 32],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = DiscretPolicy(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        action_dim=env.action_space.nvec[0] * 3,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()
    
    algorithm.train()

if __name__ == '__main__':
    main()

"""
TODO
1. self.z这个变量最好不要弄成nomal分布,就一个输出就行

"""