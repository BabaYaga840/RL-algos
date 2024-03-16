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
from utils.q_net import Q_network
from utils.Replay import ReplayMemory

class TarDQN():
  def __init__(self, buffer_size, action_s, state_s, epsilon, gamma, lr, tau):
    self.buffer_size=buffer_size
    self.replay_buffer=ReplayMemory(self.buffer_size,5)
    self.action_s=action_s
    self.state_s=state_s
    self.epsilon=epsilon
    self.lr=lr
    self.q_net=Q_network(self.state_s, self.action_s, self.lr, 0)
    self.q_tar=Q_network(self.state_s, self.action_s, self.lr, 1)
    self.q_tar.load_state_dict(self.q_net.state_dict())
    self.gamma=gamma
    self.tau=tau


  def get_action(self,s):
    s = np.array(s)
    s = torch.from_numpy(s).float()
    if np.random.rand(1)<self.epsilon:
      return np.random.rand(self.action_s)
    else:
      with torch.no_grad():
        return self.q_tar(s)

  def add_to_buffer(self, state, action, reward, next_state, done):
     self.replay_buffer.push([state, action, reward, next_state, done])

  def replay(self, batch_size):
    if len(self.replay_buffer) < batch_size:
      return
    minibatch = self.replay_buffer.sample(batch_size)
    states = np.array([s.flatten() for s, a, r, ns, d in minibatch])
    next_states = np.array([ns.flatten() for s, a, r, ns, d in minibatch])
    actions = torch.tensor([a for s, a, r, ns, d in minibatch])
    rewards = torch.tensor([r for s, a, r, ns, d in minibatch], dtype=torch.float32)
    dones = torch.tensor([d for s, a, r, ns, d in minibatch], dtype=torch.float32)
    states = torch.from_numpy(states).float()
    next_states = torch.from_numpy(next_states).float()
    current_q_values = self.q_net(states)
    next_q_values = self.q_net(next_states).detach()

    max_next_q_values = next_q_values.max(1)[0]  # Get max Q value for next states
    target_q_values = current_q_values.clone()
    for idx in range(batch_size):
        target_q_values[idx, actions[idx]] = rewards[idx] + self.gamma * max_next_q_values[idx] * (1 - dones[idx])
    
    self.q_net.optimize(current_q_values, target_q_values)
    for online_param, target_param in zip(self.q_net.parameters(), self.q_tar.parameters()):
        target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

