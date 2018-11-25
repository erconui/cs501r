import gym
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from functools import reduce
from tqdm import tqdm
import random
from time import sleep

# assert torch.cuda.is_available() # You need to request a GPU from Runtime > Change Runtime Type
class PolicyNetwork(nn.Module):
    def __init__(self, state_dimension, action_dimension):
      super(PolicyNetwork, self).__init__()
      self.policy_net = nn.Sequential(nn.Linear(state_dimension, 10),
                                     nn.ReLU(),
                                     nn.Linear(10,10),
                                     nn.ReLU(),
                                     nn.Linear(10,10),
                                     nn.ReLU(),
                                     nn.Linear(10, action_dimension))
      self.policy_softmax = nn.Softmax(dim=1)

    def forward(self, x):
      scores = self.policy_net(x)
      return self.policy_softmax(scores)


class ValueNetwork(nn.Module):
    def __init__(self, state_dimension):
      super(ValueNetwork, self).__init__()
      self.value_net = nn.Sequential(nn.Linear(state_dimension, 10),
                                     nn.ReLU(),
                                     nn.Linear(10,10),
                                     nn.ReLU(),
                                     nn.Linear(10,10),
                                     nn.ReLU(),
                                     nn.Linear(10, 1))
    def forward(self,x):
      return self.value_net(x)

class AdvantageDataset(Dataset):
    def __init__(self, experience):
        super(AdvantageDataset, self).__init__()
        self._exp = experience
        self._num_runs = len(experience)
        self._length = reduce(lambda acc, x: acc + len(x), experience, 0)

    def __getitem__(self, index):
        idx = 0
        seen_data = 0
        current_exp = self._exp[0]
        while seen_data + len(current_exp) - 1 < index:
            seen_data += len(current_exp)
            idx += 1
            current_exp = self._exp[idx]
        chosen_exp = current_exp[index - seen_data]
        return chosen_exp[0], chosen_exp[4]

    def __len__(self):
        return self._length


class PolicyDataset(Dataset):
    def __init__(self, experience):
        super(PolicyDataset, self).__init__()
        self._exp = experience
        self._num_runs = len(experience)
        self._length = reduce(lambda acc, x: acc + len(x), experience, 0)

    def __getitem__(self, index):
        idx = 0
        seen_data = 0
        current_exp = self._exp[0]
        while seen_data + len(current_exp) - 1 < index:
            seen_data += len(current_exp)
            idx += 1
            current_exp = self._exp[idx]
        chosen_exp = current_exp[index - seen_data]
        return chosen_exp

    def __len__(self):
        return self._length

def calculate_returns(trajectories, gamma):
  for i, trajectory in enumerate(trajectories):
    current_reward = 0
    for j in reversed(range(len(trajectory))):
      state, probs, action_index, reward = trajectory[j]
      ret = reward + gamma*current_reward
      trajectories[i][j] = (state, probs, action_index, reward, ret)
      current_reward = ret

def calculate_advantages(trajectories, value_net):
  for i, traj in enumerate(trajectories):
    for j, exp in enumerate(traj):#experience
      advantage = exp[4] - value_net(torch.from_numpy(exp[0]).float().unsqueeze(0))[0,0].detach().double()
      # advantage = exp[4] - value_net(exp[0].detach().numpy().float().unsqueeze(0))[0,0]
      trajectories[i][j] = (exp[0], exp[1], exp[2], exp[3], exp[4], advantage)

policy = torch.load("policy")
value = torch.load("value")
states = 8
actions = 4
env = gym.make('LunarLander-v2')
for i in range(5):
    s = env.reset()
    done = False
    while not done:
        env.render()
        action = policy(torch.from_numpy(s).float().view(1,-1))
        action_index = np.random.choice(range(actions),p=action.detach().numpy().reshape((actions)))
        s, r, done, _ = env.step(action_index)
        sleep(.01)
