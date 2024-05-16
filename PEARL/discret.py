import numpy as np
from torch import nn
import torch
from rlkit.torch.networks import Mlp
from torch.nn import functional as F
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import PyTorchModule
from rlkit.torch.core import np_ify


class DiscretPolicy(PyTorchModule, ExplorationPolicy):
    def __init__(
            self,
            obs_dim,
            latent_dim,
            action_dim,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # network
        self.net1_fc1 = nn.Linear(obs_dim // 3 + latent_dim, 128)
        self.net1_fc2 = nn.Linear(128, 128)
        self.net1_fc3 = nn.Linear(128, action_dim // 3) #TODO 注意这个地方是直接除三了
        self.net1_probs_layer = torch.nn.Softmax(dim=1)

        self.net2_fc1 = nn.Linear(obs_dim // 3 + latent_dim, 128)
        self.net2_fc2 = nn.Linear(128, 128)
        self.net2_fc3 = nn.Linear(128, action_dim // 3)
        self.net2_probs_layer = torch.nn.Softmax(dim=1)

        self.net3_fc1 = nn.Linear(obs_dim // 3 + latent_dim, 128)
        self.net3_fc2 = nn.Linear(128, 128)
        self.net3_fc3 = nn.Linear(128, action_dim // 3)
        self.net3_probs_layer = torch.nn.Softmax(dim=1)

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions, {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        probs1, probs2, probs3 = self.forward(obs, deterministic=deterministic)
        if deterministic:
            action1 = torch.argmax(probs1, dim=1).to('cpu')[0]
            action2 = torch.argmax(probs2, dim=1).to('cpu')[0]
            action3 = torch.argmax(probs3, dim=1).to('cpu')[0]
        else:
            probs1 = probs1.to('cpu').detach().numpy().reshape(-1)
            probs2 = probs2.to('cpu').detach().numpy().reshape(-1)
            probs3 = probs3.to('cpu').detach().numpy().reshape(-1)

            action1 = np.random.choice(range(self.action_dim // 3), p=probs1)
            action2 = np.random.choice(range(self.action_dim // 3), p=probs2)
            action3 = np.random.choice(range(self.action_dim // 3), p=probs3)
        action = np.array([np_ify(action1), np_ify(action2), np_ify(action3)])
        return action

    def forward(self, obs, reparameterize=False, deterministic=False, return_log_prob=False):
        output1 = obs.clone()
        output1 = torch.cat((output1[:,:self.obs_dim//3].reshape(-1,), output1[:,-self.latent_dim:].reshape(-1,))).reshape(-1, self.latent_dim+self.obs_dim//3)
        output1 = F.relu(self.net1_fc1(output1))
        output1 = F.relu(self.net1_fc2(output1))
        output1 = self.net1_fc3(output1)
        action_probabilities1 = self.net1_probs_layer(output1)

        output2 = obs.clone()
        output2 = torch.cat((output2[:,:self.obs_dim//3].reshape(-1,), output2[:,-self.latent_dim:].reshape(-1,))).reshape(-1, self.latent_dim+self.obs_dim//3)
        output2 = F.relu(self.net2_fc1(output2))
        output2 = F.relu(self.net2_fc2(output2))
        output2 = self.net2_fc3(output2)
        action_probabilities2 = self.net2_probs_layer(output2)

        output3 = obs.clone()
        output3 = torch.cat((output3[:,:self.obs_dim//3].reshape(-1,), output3[:,-self.latent_dim:].reshape(-1,))).reshape(-1, self.latent_dim+self.obs_dim//3)
        output3 = F.relu(self.net3_fc1(output3))
        output3 = F.relu(self.net3_fc2(output3))
        output3 = self.net3_fc3(output3)
        action_probabilities3 = self.net3_probs_layer(output3)

        return (action_probabilities1, action_probabilities2, action_probabilities3)