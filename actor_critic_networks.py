import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class Actor(torch.nn.Module):
    def __init__(self, obs_size, action_space, l1_size=400, l2_size=300):
        super(Actor, self).__init__()
        self.action_space = action_space

        self.layer1 = torch.nn.Linear(obs_size, l1_size)
        #bn
        self.bn1 = nn.BatchNorm1d(l1_size)
        self.layer2 = torch.nn.Linear(l1_size, l2_size)
        #bn
        self.bn2 = nn.BatchNorm1d(l2_size)
        self.layer3 = torch.nn.Linear(l2_size, action_space.shape[0])

        # Initialization and batch norm ideas from
        # https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/lunar-lander/pytorch
        f1 = 1./np.sqrt(self.layer1.weight.data.size()[0])
        nn.init.uniform_(self.layer1.weight.data, -f1, f1)
        nn.init.uniform_(self.layer1.bias.data, -f1, f1)

        f2 = 1./np.sqrt(self.layer2.weight.data.size()[0])
        nn.init.uniform_(self.layer2.weight.data, -f2, f2)
        nn.init.uniform_(self.layer2.bias.data, -f2, f2)

        f3 = 0.003 # specified in the paper
        nn.init.uniform_(self.layer3.weight.data, -f3, f3)
        nn.init.uniform_(self.layer3.bias.data, -f3, f3)

        self.action_space = action_space

    def forward(self, x):
        x = F.relu(self.layer1(x))
        #bn
        x = self.bn1(x)
        x = F.relu(self.layer2(x))
        #bn
        x = self.bn2(x)
        x = self.layer3(x) # Don't use relu on last layer!

        x = torch.tanh(x) * torch.from_numpy(self.action_space.high).float()
        return x

    def take_action(self, state, added_noise=None):

        #bn
        if state.ndim == 1:
            state = state[np.newaxis, :]

        state_x = torch.from_numpy(state).float()

        #bn
        self.eval()

        #bn
        action = self.forward(state_x).detach().numpy()[0]

        #no bn
        #action = self.forward(state_x).detach().numpy()

        if added_noise is not None:
            action += added_noise

        return action.clip(min=self.action_space.low, max=self.action_space.high) # TODO: clip action?


class Critic(torch.nn.Module):
    def __init__(self, obs_size, action_size, l1_size=400, l2_size=300):
        super(Critic, self).__init__()
        self.layer1 = torch.nn.Linear(obs_size, l1_size)
        #bn
        self.bn1 = nn.BatchNorm1d(l1_size)
        self.layer2 = torch.nn.Linear(l1_size+action_size, l2_size)
        #bn
        self.bn2 = nn.BatchNorm1d(l2_size)
        self.layer3 = torch.nn.Linear(l2_size, 1)

        # Initialization and batch norm ideas from
        # https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/lunar-lander/pytorch
        f1 = 1./np.sqrt(self.layer1.weight.data.size()[0])
        nn.init.uniform_(self.layer1.weight.data, -f1, f1)
        nn.init.uniform_(self.layer1.bias.data, -f1, f1)

        f2 = 1./np.sqrt(self.layer2.weight.data.size()[0])
        nn.init.uniform_(self.layer2.weight.data, -f2, f2)
        nn.init.uniform_(self.layer2.bias.data, -f2, f2)

        f3 = 0.0003 # specified in the paper
        nn.init.uniform_(self.layer3.weight.data, -f3, f3)
        nn.init.uniform_(self.layer3.bias.data, -f3, f3)


    def forward(self, x, a):
        layer1_out = self.layer1(x)
        #bn
        layer1_out = self.bn1(layer1_out)
        layer1_bn = F.relu(layer1_out)

        layer2_out = self.layer2(torch.cat([layer1_bn, a], dim=1))
        #bn
        layer2_out = self.bn1(layer2_out)
        layer2_bn = F.relu(layer2_out)

        q_value = self.layer3(layer2_bn)

        return q_value
