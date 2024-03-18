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

class Policy(nn.Module):
  def __init__(self, state_s, action_s, learning_rate, num=0):
    super(Policy, self).__init__()
    self.learning_rate = learning_rate
    self.layer1 = nn.Linear(state_s, 32)
    self.layera = nn.Linear(32, action_s)
    self.layerc = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.criterion = nn.MSELoss()
    self.num = num
    self.reward=[]
    self.log_prob=[]
    
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = self.layera(x)
    y = self.layerc(x)
    x = nn.functional.softmax(x, dim=0)
    return x,y


  def optimize(self,states,actions,FinalRewards,a,b):
    for state,action,G in zip(states,actions,FinalRewards):
      prob=self(torch.from_numpy(state))
      dist=Categorical(probs=prob)
      log_prob=dist.log_prob(torch.tensor(action))
      lossa = -log_prob*G
      lossc = self.criterion(a,b)
      self.optimizer.zero_grad()  
      wandb.log({f"actor loss{self.num}": lossa})
      wandb.log({f"critic loss{self.num}": lossc})
      loss = lossa+lossc
      loss.backward()  
      self.optimizer.step()  
