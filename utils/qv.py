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

class ActorCriticNetwork(nn.Module):
  def __init__(self, state_s, action_s, learning_rate, num=0, critic_weightage=1):
    super(ActorCriticNetwork, self).__init__()
    self.learning_rate = learning_rate
    self.layer1 = nn.Linear(state_s, 32)
    self.layera = nn.Linear(32, action_s)
    self.layerc = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.criterion = nn.MSELoss()
    self.num = num
    self.reward=[]
    self.log_prob=[]
    self.critic_weightage=1
    
  def forward(self, x):
    x = F.relu(self.layer1(x))
    y = self.layerc(x)
    x = self.layera(x)
    x = nn.functional.softmax(x, dim=-1)
    return x,y


  def optimize(self,states,actions,FinalRewards,delqs):
    torch.autograd.set_detect_anomaly(True)
    for state,action,G in zip(states,actions,FinalRewards):
      prob, value = self(torch.from_numpy(state))
      dist=Categorical(probs=prob)
      entropy = dist.entropy()
      log_prob=dist.log_prob(torch.tensor(action))
      
      lossa = -log_prob*G
      lossc = self.criterion(torch.tensor(delqs), torch.tensor(G))
      self.optimizer.zero_grad()  
      wandb.log({f"actor loss_actor": lossa})
      wandb.log({f"critic loss_critic": lossc})
      loss = lossa + self.critic_weightage * lossc - 0.001 * entropy
      loss.backward(retain_graph=True)  
      del prob, value, dist, entropy, log_prob, loss 
      self.optimizer.step()
    for vloss in lossc:
      del lossc

class PPONetwork(nn.Module):
  def __init__(self, state_s, action_s, learning_rate, num=0, critic_weightage=1):
    super(PPONetwork, self).__init__()
    self.learning_rate = learning_rate
    self.layer1 = nn.Linear(state_s, 32)
    self.layera = nn.Linear(32, action_s)
    self.layerc = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.criterion = nn.MSELoss()
    self.num = num
    self.reward=[]
    self.log_prob=[]
    self.critic_weightage=critic_weightage
    self.old_dist=None
    self.epsilon=0.2
    self.epochs=20
    
  def forward(self, x):
    x = F.relu(self.layer1(x))
    y = self.layerc(x)
    x = self.layera(x)
    x = nn.functional.softmax(x, dim=-1)
    return x,y


  def optimize(self,states1,actions1,FinalRewards1,vlosses1):
    for i in range(self.epochs):
      sample_indices = np.random.choice(len(states1), size=min(5,len(states1)), replace=False)
      states1=np.array(states1)
      actions1=np.array(actions1)
      FinalRewards1=np.array(FinalRewards1)
      vlosses=[vlosses1[i] for i in sample_indices]
      states=states1[sample_indices]
      actions=actions1[sample_indices]
      FinalRewards=FinalRewards1[sample_indices]
      #if self.old_dist == None:
      #  self.old_dist = Categorical(probs=self(torch.from_numpy(state)))
      for state,action,G,lossc in zip(states,actions,FinalRewards,vlosses):
        prob, value=self(torch.from_numpy(state))
        dist=Categorical(probs=prob)
        entropy = dist.entropy()
        log_prob = dist.log_prob(torch.tensor(action))
        if self.old_dist == None:
          self.old_dist = dist
        old_log_prob = self.old_dist.log_prob(torch.tensor(action))
        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        lossa = -torch.min(old_log_prob * G * ratio, old_log_prob*G * clipped_ratio)
        #lossc = self.criterion(value, G)
        self.optimizer.zero_grad()  
        wandb.log({f"actor loss_actor": lossa})
        wandb.log({f"critic loss_critic": lossc})
        loss = lossa + self.critic_weightage * lossc - 0.001 * entropy
        self.old_dist = dist
        loss.backward(retain_graph=True)
        del prob, value, dist, entropy, log_prob, old_log_prob, ratio, clipped_ratio, lossa, loss
        self.optimizer.step()
    for vloss in vlosses1:
      del vloss
    for action in actions:
      del actions

