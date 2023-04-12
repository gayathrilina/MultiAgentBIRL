#-------------------------------------------------------------------------------#
# Project: Runner script for BIRL algorithm on a domain
# Version: 21.12.01
# Last modified: 12/14/2021
# Author: Gayathri Anil, Aditya Shinde
# Reference: https://github.com/amsterg/birl
#-------------------------------------------------------------------------------#

from birl import Birl
from env import FrozenLakeEnv
from dp import DP
import sys
import gym
import os
from time import sleep
from gym.envs.toy_text import discrete
from gym import utils
from six import StringIO, b
from contextlib import closing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)


# Import environment
env = FrozenLakeEnv(is_slippery=True)
env.num_actions = env.nA #remove
env.num_states = env.nS #remove
obs = env.reset()
dp = DP(env)

# Original rewards of the environment
rewards_implicit = np.array([sum([val[2] for a in env.P[s] for val in env.P[s][a]]) for s in env.P])

# Policy for expert agent
# why 100 times
for _ in range(100):
    dp.policy_eval()
    dp.policy_imp()
dp.q_values = np.array([dp.q_values[s] for s in dp.q_values])

# Initiliase Birl agent
birl = Birl(env.num_states)

# Step 1: Generate expert trajectory
print("Running Sim")
birl.trajectory = birl.getTrajectory(dp)
print("Running Sim Done")

# Step 2: Sample a random reward function to start with
random_rewards = birl.sample_random_rewards(birl.num_states, 1, 1)

# Step 3: Run PolicyWalk to sample distribution and get the reward function for the problem
birl.policy_walk(random_rewards)

# Regenerating the environment object with the recovered rewards
env_gen_rews = FrozenLakeEnv(is_slippery=True,rewards=birl.rewards_recovered)

env_gen_rews.num_actions = env_gen_rews.nA
env_gen_rews.num_states = env_gen_rews.nS
obs = env_gen_rews.reset()
dp_rg = DP(env_gen_rews)
os.system('clear')
sleep(1)

# Step 4: Get policy for new reward function
for _ in range(100):
        dp_rg.policy_eval()
        dp_rg.policy_imp()

# Step 5: Generating trajectory for the observing agent using the learned reward function
obs_birl = Birl(env_gen_rews.num_states)
obs_birl.getTrajectory(dp_rg)

# Comparing learned reward function with original reward function
plt.figure(figsize=(8, 8), num="reward_original")
sns.heatmap(rewards_implicit.reshape(4, 4),
            cmap="Spectral", annot=True, cbar=False)
plt.figure(figsize=(8, 8), num="reward_recovered")
sns.heatmap(birl.rewards_recovered.reshape(4, 4),
            cmap="Spectral", annot=True, cbar=False)
plt.show()

# Comparing value function with learned reward function vs original
plt.figure(figsize=(8, 8), num="dp_sv")
sns.heatmap(dp.state_values.reshape(4, 4)/(np.max(dp.state_values)-np.min(dp.state_values)),
            cmap="Spectral", annot=True, cbar=False)
plt.figure(figsize=(8, 8), num="gen_r_dp_sv")
sns.heatmap(dp_rg.state_values.reshape(4, 4)/np.max(dp_rg.state_values)-np.min(dp_rg.state_values),
            cmap="Spectral", annot=True, cbar=False)
plt.show()



