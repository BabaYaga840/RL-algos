import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import wandb

class Q_network(nn.Module):
  def __init__(self, state_s, action_s, learning_rate, num=0):
    super(Q_network, self).__init__()
    self.learning_rate = learning_rate
    self.Q_sa = np.zeros((state_s,action_s))
    self.layer1 = nn.Linear(state_s, 128)
    self.layer2 = nn.Linear(128, 128)
    self.layer3 = nn.Linear(128, action_s)
    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.criterion = nn.MSELoss()
    self.num = num
  
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return self.layer3(x)

  def optimize(self,a,b):
    self.optimizer.zero_grad()  
    loss = self.criterion(a,b)  
    wandb.log({f"loss{self.num}": loss})
    loss.backward()  
    self.optimizer.step()  
