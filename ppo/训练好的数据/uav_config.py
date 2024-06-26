#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

uav_config = {
    ## Commented parameters are set to default values in ppo

    #==========  env config ==========
    'env': 'PongNoFrameskip-v4',  # environment name
    'continuous_action': False,  # action type of the environment
    'env_num': 1,  # number of the environment
    'seed': None,  # seed of the experiment
    'xparl_addr': None,  # xparl address for distributed training

    #==========  training config ==========
    'train_total_steps': int(1e7),  # max training steps
    'step_nums': 32,  # data collecting time steps (ie. T in the paper)
    'num_minibatches': 2,  # number of training minibatches per update.
    'update_epochs': 4,  # number of epochs for updating (ie K in the paper)
    'eval_episode': 3,
    'test_every_steps': int(5e3),  # interval between evaluations

    #========== coefficient of ppo ==========
    'initial_lr': 2.5e-4,  # start learning rate
    'lr_decay': True,  # whether or not to use linear decay rl
    # 'eps': 1e-5,  # Adam optimizer epsilon (default: 1e-5)
    'clip_param': 0.1,  # epsilon in clipping loss
    'entropy_coef': 0.01,  # Entropy coefficient (ie. c_2 in the paper)
    # 'value_loss_coef': 0.5,  # Value loss coefficient (ie. c_1 in the paper)
    # 'max_grad_norm': 0.5,  # Max gradient norm for gradient clipping
    # 'use_clipped_value_loss': True,  # advantages normalization
    # 'clip_vloss': True,  # whether or not to use a clipped loss for the value function
    # 'gamma': 0.99, # discounting factor
    # 'gae': True,  # whether or not to use GAE
    # 'gae_lambda': 0.95,  # Lambda parameter for calculating N-step advantage
}
