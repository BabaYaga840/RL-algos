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
from models.REINFORCE import REINFORCE


with open("config.yaml") as f:
    cfg=yaml.load(f, Loader=yaml.FullLoader)

# PARAMETERS
environ=cfg["env"]
algo=cfg["algo"]

cfgm=cfg["model"]
num_iterations = cfgm["num_iterations"]

learning_rate = cfgm["learning_rate"]  
gamma = cfgm["gamma"]

re=[]

def RF(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, environ=environ):
    global re
    env = gym.make(environ, max_episode_steps=1000)
    eval_env = gym.make(environ)
    rewards=[]    
    agent = REINFORCE(env.action_space.n, 
                        env.observation_space.shape[0],
                        gamma=gamma,
                        lr=learning_rate)
    print("---------------------------------------------------------")
    print("running REINFORCE")
    print("---------------------------------------------------------")
    observation = env.reset(seed=42)
    for iteration in range(n_timesteps):
        state = env.reset()
        done = False
        total_reward=0
        rewards=[]
        actions=[]
        states=[]
        while not done:
          action = agent.get_action(state)
          next_state, reward, done, truncated = env.step(int(action))
          actions.append(action)
          rewards.append(reward)
          states.append(state)
          total_reward += reward
          state=next_state
        env.close()
        wandb.log({"rewards": total_reward})


        FinalRewards=[]
        for i in range(len(rewards)):
            G=0.0
            for j,r in enumerate(rewards[i:]):
                G +=r*(gamma**j)
            FinalRewards.append(G)
        agent.policy_net.optimize(states,actions,FinalRewards)
    return rewards



if __name__ == "__main__":
    wandb.init(
    project="my-awesome-project",
    config={
        "environment": environ,
        "Algorithm": algo,
    "num_iterations": num_iterations,
    "learning_rate": learning_rate,
    "gamma": gamma,
    }
    )    
    rewards=RF()
    wandb.finish()



    
