from copy import deepcopy
import numpy


class DP:
    def __init__(self, env, gamma=0.8):
        self.env = env
        self.gamma = gamma

        self.state_values = numpy.ones(
            self.env.num_states)*100/self.env.num_states
        self.policy = numpy.zeros(
            [self.env.num_states, self.env.num_actions])/self.env.num_actions
        self.q_values = {s: numpy.ones(self.env.num_actions)
                         for s in range(self.env.num_states)}

    def policy_eval(self):
        while True:
            delta = 0.0
            delta_thres = 1e-5

            for s in range(self.env.num_states):
                sv = 0
                for a, ap in enumerate(self.policy[s]):
                    for s_p in range(self.env.num_states):
                        sv += ap * self.env.T[s, a, s_p] * \
                                (self.env.R[s, a] + \
                                self.gamma * self.state_values[s_p])
 
                delta = max(delta, numpy.abs(sv-self.state_values[s]))

                self.state_values[s] = sv
            
            if delta < delta_thres:
                break

    def policy_imp(self):
        policy_stable = True
        for s in range(self.env.num_states):
            curr_action = numpy.argmax(self.policy[s])
            action_vals = numpy.zeros(self.env.num_actions)
            for a in range(self.env.num_actions):
                for s_p in range(self.env.num_states):
                    av = self.env.T[s, a, s_p] * (self.env.R[s, a] + \
                            self.gamma * self.state_values[s_p])
                    action_vals[a] += av
                
            self.q_values[s] = action_vals
            action_best = numpy.argmax(action_vals)
            if action_best != curr_action:
                policy_stable = False
            self.policy[s] = numpy.eye(self.env.num_actions)[action_best]

    def policy_iteration(self):
        it = 0
        while True:
            it += 1
            print("Policy iteration - iteration #", it)
            old_policy = deepcopy(self.policy)
            self.policy_eval()
            self.policy_imp()
            if numpy.all(old_policy == self.policy):
                break

