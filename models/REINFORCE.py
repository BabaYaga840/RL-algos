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
from utils.policy_net import Policy

class REINFORCE():
  def __init__(self, action_s, state_s, gamma, lr):
    self.action_s=action_s
    self.state_s=state_s
    self.lr=lr
    self.poicy_net=Policy(self.state_s, self.action_s, self.lr)
    self.gamma=gamma


  def get_action(self,s):
    s = np.array(s)
    s = torch.from_numpy(s).float()
    probs = self.poicy_net(s)
    m = Categorical(probs)
    action = m.sample()
    return action.item()
