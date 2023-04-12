#-------------------------------------------------------------------------------#
# Project: Bayesian Inverse Reinforcement Learning Algorithm
# Version: 21.12.01
# Last modified: 12/14/2021
# Author: Gayathri Anil, Aditya Shinde
# Reference: https://github.com/amsterg/birl
#-------------------------------------------------------------------------------#

import copy
from dp import DP
import matplotlib.pyplot as plt
import seaborn as sns
from env import FrozenLakeEnv
import numpy as np
np.random.seed(42)
import os
from time import sleep
import sys
sys.path.append("../env/")
from lb_foraging import ForagingEnv
import prob_dists

num_samples = 200

class Birl():

    def __init__(self, num_states, domain):
        self.num_states = num_states
        self.gamma = 0.8
        self.alpha = 10
        self.trajectory = None
        self.rewards_recovered = None
        self.num_row, self.num_col, self.num_agent, self.num_food = domain

    # Function to sample reward randomly
    def sample_random_rewards(self, n_states, dist, step_size=None, r_max=None, r_min=None, mu = None, sigma = None):
        """
        sample random rewards form gridpoint(R^{n_states}/step_size).
        :param n_states:
        :param step_size:
        :param r_max:
        :return: sampled rewards
        """
        if dist == "uniform":
            rewards = np.random.uniform(low=r_min, high=r_max, size=n_states)
        elif dist == "gaussian":
            rewards = np.random.normal(mu, sigma, n_states)


        # move these random rewards toward a gridpoint add r_max to makee mod to be always positive add step_size for easier clipping
        #rewards = rewards + r_max + step_size
        rewards = rewards + step_size

        for i, reward in enumerate(rewards):
            mod = reward % step_size
            rewards[i] = reward - mod

        # subtract added values from rewards
        #rewards = rewards - (r_max + step_size)
        rewards = rewards - (step_size)
        return rewards

    # Function to sample reward using mcmc
    def mcmc_reward_step(self, rewards, step_size, r_max, r_min):
        new_rewards = copy.deepcopy(rewards)
        index = np.random.randint(len(rewards))
        step = np.random.choice([-step_size, step_size])
        new_rewards[index] += step
        new_rewards = np.clip(a=new_rewards, a_min=r_min, a_max=r_max)
        if np.all(new_rewards == rewards):
            new_rewards[index] -= step
        assert np.any(rewards != new_rewards), 'rewards do not change: {}, {}'.format(
            new_rewards, rewards)
        return new_rewards
    
    # Function to check if the Q function of new policy is optimal
    def optimal_q_check(self, q_values, pi):
        assert q_values.shape == pi.shape, "Shapes mismatch for qvalues in qs_comp"
        for s in range(q_values.shape[0]):
            for a in range(q_values.shape[1]):
                if q_values[s, a] > q_values[s, np.argmax(pi[s])]:
                    # if atleast one (s,a) exists that is to be optimizied(kinda)
                    return True
        return False

    # Function to get the posterior probability of sampled reeward point
    def posterior(self, agent_with_env, prior):
        agent_with_env.policy_imp()
        q_vals = agent_with_env.q_values
        env = agent_with_env.env
        # for s, a in self.trajectory:
        #     print("s = ", s, " a = ", a)
        #     print("s map = ", env.S_map[s])
        #     print("q value = ", q_vals[env.S_map[s]][a])
        #     print("q values = ", q_vals[env.S_map[s]])

        value = np.sum([self.alpha * q_vals[env.S_map[s]][a] - np.log(np.sum(np.exp(self.alpha * q_vals[env.S_map[s]]))) for s, a in self.trajectory]) + np.log(prior(env.rewards))
        #print("val -----> ", value)
        return value

    # Function to get the probability ratio to accept or reject a newly sampled point
    def posteriors_ratio(self, dp, dp_next, prior=1):
        ln_p_new = self.posterior(dp_next, prior)
        ln_p = self.posterior(dp, prior)
        return np.exp(ln_p_new - ln_p)

    # Function to perform MCMC PolicyWalk sampling method
    def policy_walk(self, original_rewards, random_rewards, file_name, prior):

        ## Step 1: Generate random starting rewards for all states using uniform distribution
        #random_rewards = self.sample_random_rewards(self.num_states, 1, 1)
        
        # Step 2: Perform policy iteration for the random starting rewards to get policy and Q function
        env = ForagingEnv(self.num_row, self.num_col, self.num_agent, self.num_food, random_rewards)
        #env.num_actions = len(env.A) #remove
        #env.num_states = len(env.S) #remove
        obs = env.reset()
        dp = DP(env)

        dp.policy_iteration()

        dp.q_values = np.array([dp.q_values[s] for s in dp.q_values])
        pi = dp.policy

        # Step 3: Sample till convergence
        for _ in range(num_samples):
            print("Sample ", _+1, " of ", num_samples, " samples")

            random_rewards = env.rewards

            # Run mcmc_reward_step to get new_rewards
            new_rewards = self.mcmc_reward_step(random_rewards, step_size=0.5, r_max=1, r_min=-1)
            
            # Perform policy iteration for new_rewards
            env_new = ForagingEnv(self.num_row, self.num_col, self.num_agent, self.num_food, new_rewards)
            env_new.rewards = new_rewards 
            #env_new = FrozenLakeEnv(is_slippery=True, rewards=new_rewards)
            env_new.num_actions = len(env_new.A) #remove
            env_new.num_states = len(env_new.S) #remove
            
            dp_next = DP(env_new)
            dp_next.policy_iteration()
            
            # Get new q values for new_rewards
            dp_new_q_values = np.array([dp_next.q_values[s]
                                        for s in dp_next.q_values])

            # New Q values and previous sample's policy is used here for the new object
            dp_next = DP(env_new)
            dp_next.policy = pi
            dp_next.q_values = dp_new_q_values

            plt.plot(range(env.num_states), original_rewards/np.linalg.norm(original_rewards), label = "Actual rewards")
            plt.plot(range(env.num_states), new_rewards/np.linalg.norm(new_rewards), label = "Learned rewards")
            plt.legend()
            plt.xlabel("States")
            plt.ylabel("Normalised Rewards")
            plt.title("PolicyWalk sampling of Rewards | Sample " + str(_+1))
            plt.savefig(file_name + "/rewards" + str(_+1) + ".png")
            #plt.show()
            plt.close()

            """
            if "dp_q_values < dp_new_q_values": (or)
            if "dp_new_q_values(pi) < dp_new_q_values" (with this for now):
            """
            # Check if the newly sampled point should be accepted or rejected
            if self.optimal_q_check(dp_next.q_values, pi):
                dp_next.policy_iteration()
                pi_new = dp_next.policy
                """
                prob_comparision = update env(rews) policy with prob ( min(1, ratio(posterioirs of dp,dp_next's policies)))
                """
                # if posteriors_ratio(env_new,pi_new,env,pi,prior,)
                if np.random.random() < self.posteriors_ratio(dp, dp_next, prior):
                    print("update env and pi")

                    # "porb comparison":
                    env, pi = env_new, pi_new
            else:
                if np.random.random() < self.posteriors_ratio(dp, dp_next, prior):
                    # if "prob comparison":
                    print("update env")

                    env = env_new
        # Return reward of the last sample point as the recovered reward function
        self.rewards_recovered = env.rewards

    # Function to get trajectory given a policy
    def getTrajectory(self, agent_with_env):
        # simulating trajectory with the help of expert policy
        done = False
        trajectory = []
        env = agent_with_env.env
        policy = agent_with_env.policy
        obs = agent_with_env.env.reset()
        ix = 0
        while True:
            ix+=1
            env.render()
            action = np.argmax(policy[env.S_map[obs]])
            trajectory.append([obs, action])
            obs, _, done, _ = env.step(action)
            done_test = tuple([(-1, -1) for i in range(env.n + env.f)])
            if done:
                if obs == done_test:
                    print("if")
                    env.render()

                    break
                else:
                    print("else")
                    env.reset()
        print(ix)
        sleep(1)
        return trajectory

    def prepare_prior(self, dist, r_max):
        prior = getattr(prob_dists, dist[0].upper() + dist[1:] + 'Dist')
        if dist == 'uniform':
            return prior(xmax=r_max)
        elif dist == 'gaussian':
            return prior()
        elif dist in {'beta', 'gamma'}:
            return prior(loc=-r_max, scale=1/(2 * r_max))
        else:
            raise NotImplementedError('{} is not implemented.'.format(dist))
