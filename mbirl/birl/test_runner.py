#-------------------------------------------------------------------------------#
# Project: Runner script for BIRL algorithm on a domain
# Version: 21.12.01
# Last modified: 12/14/2021
# Author: Gayathri Anil, Aditya Shinde
# Reference: https://github.com/amsterg/birl
#-------------------------------------------------------------------------------#

from seaborn.rcmod import reset_defaults
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
sys.path.append("../env/")
from lb_foraging import ForagingEnv
import pickle
from datetime import datetime
from scipy.spatial import distance
import prob_dists

#---------------
# User Input 
#---------------
num_row = 3
num_col = 2
num_agent = 2
num_food = 2

# choose from: "uniform", "gaussian", "beta"
prior_distribution = "uniform"

# For Uniform distribution
max_reward = 2
min_reward = -1
step_size = 1

# For Gaussian distribution
mean = 1.3
std_dev = 0.5
step_size = 1
#---------------
_domain = (num_row, num_col, num_agent, num_food)
print("Domain: ", _domain)

current_time = datetime.now().strftime("%H.%M")
file_name = "./results/" + str(num_row) + "." + str(num_col)+ "grid-" + str(num_agent) + "agent" + str(num_food) + "food-" + prior_distribution + "prior-" + current_time
os.mkdir(file_name)

# Import environment
env = ForagingEnv(num_row, num_col, num_agent, num_food)
obs = env.reset()
dp = DP(env)

print("Environment imported")
print("Number of states: ", env.num_states, " Number of actions: ", env.num_actions)

rewards_implicit = [0] * env.num_states

# Original rewards of the environment
for s in env.S:
    rewards_implicit[env.S_map[s]] = np.max(env.R[env.S_map[s]])

mean = np.mean(rewards_implicit) - 0.2
std_dev = np.std(rewards_implicit) - 0.05

print("Implicit rewards of the environment computed")

#rewards_implicit = np.array([sum([val[2] for a in env.P[s] for val in env.P[s][a]]) for s in env.P])

# Policy for expert agent
print("Policy iteration for expert agent")
dp.policy_iteration()

dp.q_values = np.array([dp.q_values[s] for s in dp.q_values])

# Initiliase Birl agent
birl = Birl(env.num_states, _domain)

# Step 1: Generate expert trajectory
exp_trajectory = birl.trajectory = birl.getTrajectory(dp) 

print("Expert trajectory generated")

# Step 2: Sample a random reward function to start with
if prior_distribution == "uniform":
    random_rewards = birl.sample_random_rewards(birl.num_states, dist = prior_distribution, step_size = step_size, r_max = max_reward, r_min = min_reward)
elif prior_distribution == "gaussian":
    random_rewards = birl.sample_random_rewards(birl.num_states, dist = prior_distribution, step_size = step_size, mu = mean, sigma = std_dev)

print("Random reward function sampled")

#Get prior dist
prior = birl.prepare_prior(prior_distribution, max_reward)

print("PolicyWalk sampling.....")
# Step 3: Run PolicyWalk to sample distribution and get the reward function for the problem
birl.policy_walk(rewards_implicit, random_rewards, file_name, prior)

print("PolicyWalk sampling completed")

# Regenerating the environment object with the recovered rewards
env_gen_rews = ForagingEnv(num_row, num_col, num_agent, num_food, birl.rewards_recovered)

obs = env_gen_rews.reset()
dp_rg = DP(env_gen_rews)
os.system('clear')
sleep(1)

# Step 4: Get policy for new reward function
print("Policy iteration for recovered reward function")
dp_rg.policy_iteration()

# Step 5: Generating trajectory for the observing agent using the learned reward function
obs_birl = Birl(env_gen_rews.num_states, _domain)
obs_trajectory = obs_birl.getTrajectory(dp_rg)

print("Expert trajectory: ", exp_trajectory)
print("Observer trajectory: ", obs_trajectory)

# Comparing learned reward function with original reward function
plt.plot(range(env.num_states), rewards_implicit, label = "Actual rewards")
plt.plot(range(env.num_states), birl.rewards_recovered, label = "Learned rewards")
plt.legend()
plt.xlabel("States")
plt.ylabel("Rewards")
plt.savefig(file_name + "/final_rewards.png")
plt.show()
plt.close()

# Comparing normalised learned reward function with normalised original reward function
plt.plot(range(env.num_states), rewards_implicit/np.linalg.norm(rewards_implicit), label = "Actual rewards")
plt.plot(range(env.num_states), birl.rewards_recovered/np.linalg.norm(birl.rewards_recovered), label = "Learned rewards")
plt.legend()
plt.xlabel("States")
plt.ylabel("Normalised Rewards")
plt.savefig(file_name + "/final_rewards_normalised.png")
plt.show()
plt.close()

# Comparing value function with learned reward function vs original
plt.plot(range(env.num_states), dp.state_values, label = "Actual values")
plt.plot(range(env.num_states), dp_rg.state_values, label = "Learned values")
plt.legend()
plt.xlabel("States")
plt.ylabel("State Values")
plt.savefig(file_name + "/state-values.png")
plt.show()
plt.close()

# Saving results to local
results_obj = {}
results_obj['expert_q'] = dp.q_values
results_obj['learner_q'] = dp_rg.q_values
results_obj['expert_value'] = dp.state_values
results_obj['learner_value'] = dp_rg.state_values
results_obj['expert_trajectory'] = exp_trajectory
results_obj['learner_trajectory'] = obs_trajectory
results_obj['expert_rewards'] = rewards_implicit
results_obj['learner_rewards'] = birl.rewards_recovered
results_obj['euclidean_distance'] = distance.euclidean(rewards_implicit/np.linalg.norm(rewards_implicit), birl.rewards_recovered/np.linalg.norm(birl.rewards_recovered))

pickle_file = open(file_name + "/results.pickle", "wb")
pickle.dump(results_obj, pickle_file)
pickle_file.close()
