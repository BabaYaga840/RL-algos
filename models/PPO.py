import random
from collections import namedtuple, deque
from torch.distributions import Categorical
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import wandb
from utils.qv import PPONetwork


class PPO():
  def __init__(self, action_s, state_s, gamma, lr, critic_weightage):
    self.action_s=action_s
    self.state_s=state_s
    self.lr=lr
    self.critic_weightage = critic_weightage
    self.policy_net=PPONetwork(self.state_s, self.action_s, self.lr, critic_weightage = self.critic_weightage)
    self.gamma=gamma


  def get(self,s):
    s = np.array(s)
    s = torch.from_numpy(s).float()
    probs, value = self.policy_net(s)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), value

  def get_qval(self,s):
    s = np.array(s)
    s = torch.from_numpy(s).float()
    _,a = self.policy_net(s)
    return a
      
