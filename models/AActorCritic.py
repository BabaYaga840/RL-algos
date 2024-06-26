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
from utils.qv import ActorCriticNetwork


class AActorCritic():
  def __init__(self, action_s, state_s, gamma, lr, critic_weightage):
    self.device="cpu"
    self.action_s=action_s
    self.state_s=state_s
    self.lr=lr
    self.critic_weightage = critic_weightage
    self.policy_net=ActorCriticNetwork(self.state_s, self.action_s, self.lr, critic_weightage = self.critic_weightage)
    self.gamma=gamma
    self.policy_net.to(self.device)


  def get(self,s):
    s = np.array(s)
    s = torch.from_numpy(s).float()
    probs, value = self.policy_net(s.to(self.device))
    m = Categorical(probs)
    action = m.sample()
    log_prob=m.log_prob(torch.tensor(action))
    entropy=m.entropy().mean()
    return action.item(), value, log_prob, entropy

  def get_qval(self,s):
    s = np.array(s)
    s = torch.from_numpy(s).float()
    _,a = self.policy_net(s.to(self.device))
    return a
      
