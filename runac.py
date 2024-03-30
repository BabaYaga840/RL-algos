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
from models.AC import A2C
from models.AActorCritic import AActorCritic
from models.PPO import PPO



with open("config.yaml") as f:
    cfg=yaml.load(f, Loader=yaml.FullLoader)

# PARAMETERS
environ=cfg["env"]
algo=cfg["algo"]

cfgm=cfg["model"]
num_iterations = cfgm["num_iterations"]

learning_rate = cfgm["learning_rate"]  
gamma = cfgm["gamma"]
critic_weightage = cfgm["critic_weightage"]

re=[]

def run_ActorCritic(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, environ=environ):
    global re
    env = gym.make(environ, max_episode_steps=1000)
    eval_env = gym.make(environ)
    rewards=[]    
    agent = A2C(env.action_space.n, 
                        env.observation_space.shape[0],
                        gamma=gamma,
                        lr=learning_rate)
    print("---------------------------------------------------------")
    print("running basic actor critic")
    print("---------------------------------------------------------")
    observation = env.reset(seed=42)
    for iteration in range(n_timesteps):
        state = env.reset()
        done = False
        total_reward=0
        rewards=[]
        actions=[]
        states=[]
        delqs=[]
        while not done:
          action = agent.get_action(state)
          next_state, reward, done, truncated = env.step(int(action))
          reward = reward
          delq = agent.get_qval(state)
          actions.append(action)
          rewards.append(reward)
          states.append(state)
          delqs.append(delq)
          if done:
              delqs.append(agent.get_qval(next_state))
          total_reward += reward
          state=next_state
        env.close()
        wandb.log({"rewards": total_reward})


        FinalRewards=[]
        value_loss=torch.tensor(0)
        for i in range(len(rewards)):
            G=0.0
            for j,r in enumerate(rewards[i:]):
                G +=r*(gamma**j)
                value_loss = value_loss + (delqs[i]-G)**2
                #agent.q_net.optimize(torch.tensor(G),delqs[i])
                G=G+delqs[i+1]-delqs[i]
            FinalRewards.append(G)
        agent.policy_net.optimize(states,actions,rewards)
        value_loss.backward()
    return rewards


def run_ActorCritic2(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, environ=environ):
    global re
    env = gym.make(environ, max_episode_steps=1000)
    eval_env = gym.make(environ)
    rewards=[]    
    agent = AActorCritic(env.action_space.n, 
                        env.observation_space.shape[0],
                        gamma=gamma,
                        lr=learning_rate,
                        critic_weightage = critic_weightage)
    print("---------------------------------------------------------")
    print("running advantage actor critic with entropy")
    print("---------------------------------------------------------")
    observation = env.reset(seed=42)
    for iteration in range(n_timesteps):
        state = env.reset()
        done = False
        total_reward=0
        entropy=0
        rewards=[]
        actions=[]
        states=[]
        delqs=[]
        while not done:
          action, delq = agent.get(state)
          #entropy += entropy + action.entropy().mean()
          next_state, reward, done, truncated = env.step(int(action))
          reward = reward
          actions.append(action)
          rewards.append(reward)
          states.append(state)
          delqs.append(delq)
          if done:
              delqs.append(agent.get_qval(next_state))
          total_reward += reward
          state=next_state
        env.close()
        wandb.log({"rewards": total_reward})
        wandb.log({"steps": iteration})


        FinalRewards=[]
        value_loss=torch.tensor(0)
        for i in range(len(rewards)):
            G=0.0
            for j,r in enumerate(rewards[i:]):
                G +=r*(gamma**j)
                value_loss = value_loss + (delqs[i]-G)**2
                #agent.q_net.optimize(torch.tensor(G),delqs[i])
                G_next = delqs[i + 1].clone().detach()
                G = G + G_next - delqs[i].detach()
            FinalRewards.append(G)
        agent.policy_net.optimize(states,actions, FinalRewards,delqs)
    return rewards


def run_PPO(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, environ=environ):
    global re
    env = gym.make(environ, max_episode_steps=1000)
    eval_env = gym.make(environ)
    rewards=[]    
    agent = PPO(env.action_space.n, 
                        env.observation_space.shape[0],
                        gamma=gamma,
                        lr=learning_rate,
                        critic_weightage = 1)
    print("---------------------------------------------------------")
    print("running PPO")
    print("---------------------------------------------------------")
    observation = env.reset(seed=42)
    for iteration in range(n_timesteps):
        state = env.reset()
        done = False
        total_reward=0
        entropy=0
        rewards=[]
        actions=[]
        states=[]
        delqs=[]
        while not done:
          action, delq = agent.get(state)
          #entropy += entropy + action.entropy().mean()
          next_state, reward, done, truncated = env.step(int(action))
          reward = reward
          actions.append(action)
          rewards.append(reward)
          states.append(state)
          delqs.append(delq)
          if done:
              delqs.append(agent.get_qval(next_state))
          total_reward += reward
          state=next_state
        env.close()
        wandb.log({"rewards": total_reward})
        wandb.log({"steps": iteration})


        FinalRewards=[]
        value_loss=torch.tensor(0)
        value_losses=[]
        for i in range(len(rewards)):
            G=0.0
            for j,r in enumerate(rewards[i:]):
                G +=r*(gamma**j)
                #value_loss = value_loss + (delqs[i]-G)**2
                #agent.q_net.optimize(torch.tensor(G),delqs[i])
                G_next = delqs[i + 1].clone().detach()
                G = G + G_next - delqs[i].detach()
            #value_losses.append(value_loss)
            FinalRewards.append(G)
        agent.policy_net.optimize(states,actions,rewards,value_losses)
    return rewards




if __name__ == "__main__":
    wandb.init(
    project="RL-algos",
    config={
        "environment": environ,
        "Algorithm": algo,
    "num_iterations": num_iterations,
    "learning_rate": learning_rate,
    "gamma": gamma,
    }
    )
    if algo == "AC_Base":
        rewards=run_ActorCritic()
    elif algo == "AC":
        rewards=run_ActorCritic2()
    else:
        rewards=run_PPO()
    
    wandb.finish()



    
