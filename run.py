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
import yaml
from utils.q_net import Q_network
from utils.Replay import ReplayMemory
from models.DQN import DQN
from models.dDQN import dDQN


with open("config.yaml") as f:
    cfg=yaml.load(f, Loader=yaml.FullLoader)

# PARAMETERS
environ=cfg["env"]
algo=cfg["algo"]

cfgm=cfg["model"]
num_iterations = cfgm["num_iterations"]

replay_buffer_max_length = cfgm["replay_buffer_max_length"]

batch_size = cfgm["batch_size"]  

learning_rate = cfgm["learning_rate"]  
gamma = cfgm["gamma"]
epsilon = cfgm["epsilon"]

re=[]

def dqn(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, policy="egreedy", epsilon=epsilon, environ=environ):
    global re
    env = gym.make(environ, max_episode_steps=1000)
    eval_env = gym.make(environ)
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

def ddqn(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, policy="egreedy", epsilon=epsilon, environ=environ):
    global re
    env = gym.make(environ, max_episode_steps=1000)
    eval_env = gym.make(environ)
    rewards=[]
    agent = dDQN(replay_buffer_max_length,
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
          ch=np.random.randint(2)
          action = agent.get_action(state,ch).argmax()
          next_state, reward, done, truncated = env.step(int(action))
          total_reward=total_reward+reward
          agent.add_to_buffer(state,action,reward,next_state,done,ch)
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
        "environment": environ,
        "Algorithm": algo,
    "num_iterations": num_iterations,
    "replay_buffer_max_length": replay_buffer_max_length,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "epsilon": epsilon
    }
    )
    if algo=="dqn":
        rewards=dqn()
    else:
        rewards=ddqn()
    wandb.finish()



    
