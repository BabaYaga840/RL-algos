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
from models.DQN import DQN


# PARAMETERS 
num_iterations = 1000 

replay_buffer_max_length = 10000

batch_size = 64  

learning_rate = 1e-1  
gamma = 0.99
epsilon = 0.05

re=[]

def dqn(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, policy="egreedy", epsilon=epsilon):
    global re
    env = gym.make("CartPole-v1", max_episode_steps=1000)
    eval_env = gym.make("CartPole-v1")
    rewards=[]
    agent = DQN(replay_buffer_max_length,
                        env.action_space.n, 
                        env.observation_space.shape[0],
                        epsilon=epsilon, 
                        gamma=gamma,
                        lr=learning_rate)
    observation = env.reset(seed=42)
    for iteration in range(n_timesteps):
        state = env.reset()
        done = False
        total_reward=0
        while not done:
          action = agent.get_action(state).argmax()
          next_state, reward, done, truncated = env.step(int(action))
          total_reward=total_reward+reward
          agent.add_to_buffer(state,action,reward,next_state,done)
          if len(agent.replay_buffer)>=batch_size:
            re=agent.replay_buffer
            agent.replay(batch_size)
          if done:
            rewards.append(total_reward)
            wandb.log({"total_reward": total_reward})
            total_reward=0
            break
          state=next_state
        env.close()
    return rewards

if __name__ == "__main__":
    wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
        "environment": "Cartpole",
        "Algorithm": "DQN",
    "num_iterations": num_iterations,
    "replay_buffer_max_length": replay_buffer_max_length,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "epsilon": epsilon
    }
    )
    rewards=dqn()
    wandb.finish()



    
