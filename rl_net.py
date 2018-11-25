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


def main(env, policy, value, policy_optim, value_optim, actions, epochs = 300):


    # ... more stuff here...
    value_criteria = nn.MSELoss()

    # Hyperparameters
    env_samples = 100
    episode_length = 2000
    gamma = 0.99
    value_epochs = 2
    policy_epochs = 10
    batch_size = 32
    policy_batch_size = 256
    epsilon = 0.2
    # standing_time_list = []
    max_x_list = []
    loss_list = []
    reward_list = []

    loop = tqdm(total=epochs, position=0, leave=False)

    for epoch in range(epochs):
        # generate rollouts
        rollouts = []
        # standing_length = 0
        # max_x_total = -1.2
        total_reward = 0
        for _ in range(env_samples):
            # don't forget to reset the environment at the beginning of each episode!
            # rollout for a certain number of steps!
            current_rollout = []
            s = env.reset()
            s_start = s
            reward = 0
            done = False
            while not done:
            #for i in range(episode_length):
              action = policy(torch.from_numpy(s).float().view(1,-1))
              action_index = np.random.choice(range(actions),p=action.detach().numpy().reshape((actions)))
              s_prime, r, done, _= env.step(action_index)
              reward += r

              current_rollout.append((s, action.detach().reshape(-1), action_index, r))

              s = s_prime
            rollouts.append(current_rollout)
            # print(s_start[0], max_x)
            # max_x_total += max_x
            total_reward += reward

        # avg_standing_time = standing_length / env_samples
        # avg_max_x = max_x_total / env_samples
        avg_reward = total_reward / env_samples
        # standing_time_list.append(avg_standing_time)
        #max_x_list.append(avg_reward)
        reward_list.append(avg_reward)
        # print('avg standing time:', standing_length / env_samples)

        calculate_returns(rollouts, gamma)

        # Approximate the value function
        value_dataset = AdvantageDataset(rollouts)
        value_loader = DataLoader(value_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        for _ in range(value_epochs):
            # train value network
            total_loss = 0
            for state, returns in value_loader:
              value_optim.zero_grad()
              returns = returns.unsqueeze(1).float()
              expected_returns = value(state.float())
              loss = value_criteria(expected_returns, returns)
              total_loss += loss.item()
              loss.backward()
              value_optim.step()
            loss_list.append(total_loss)

        calculate_advantages(rollouts, value)

        # Learn a policy
        policy_dataset = PolicyDataset(rollouts)
        policy_loader = DataLoader(policy_dataset, batch_size=policy_batch_size, shuffle=True, pin_memory=True)
        for _ in range(policy_epochs):
            # train policy network
            for state, probs, action_index, reward, ret, advantage in policy_loader:
              policy_optim.zero_grad()
              current_batch_size = reward.size()[0]
              advantage = advantage.detach().float()#ret.float()
              p = policy(state.float())
              ratio = p[range(current_batch_size), action_index] / probs[range(current_batch_size), action_index]

              lhs = ratio*advantage
              rhs = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage
              loss = -torch.mean(torch.min(lhs, rhs))
              loss.backward()
              policy_optim.step()
        # loop.set_description('standing time: {}'.format(avg_standing_time))
        if epoch % 5 == 0 and epoch != 0:
            if epoch % 10 == 0:
                torch.save(policy, 'policy-{}'.format(epoch))
                torch.save(value, 'value-{}'.format(epoch))
            loop.set_description('Reward: {}'.format(avg_reward))
            loop.update(5)
    return env, policy, reward_list, loss_list

states = 8
actions = 4
if False:
    # env = gym.make('LunarLander-v2')
    policy = PolicyNetwork(states, actions)
    value = ValueNetwork(states)
    # policy_optim = optim.Adam(policy.parameters(), lr=1e-4, weight_decay=0.01)
    # value_optim = optim.Adam(value.parameters(), lr=1e-3, weight_decay=1)
else:
    policy = torch.load("policy")
    value = torch.load("value")

# states = 8
# actions = 4
env = gym.make('LunarLander-v2')
policy_optim = optim.Adam(policy.parameters(), lr=1e-4, weight_decay=0.01)
value_optim = optim.Adam(value.parameters(), lr=1e-3, weight_decay=1)
epochs = 900

env, policy, reward, loss = main(env, policy, value, policy_optim, value_optim, actions, epochs)
