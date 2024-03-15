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

class ReplayMemory():

    def __init__(self, capacity, size):
        self.memory = deque([], maxlen=capacity)
        self.size = size

    def push(self, list):
        assert len(list)==self.size
        self.memory.append(list)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
