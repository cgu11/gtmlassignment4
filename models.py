from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning

import pandas as pd
import numpy as np
import time as time

def run_value_iteration(P, R, gamma=0.9, max_iter=1e4, epsilon=0.0001):
    vi = ValueIteration(P, R, gamma, epsilon=epsilon, max_iter=max_iter)
    results = pd.DataFrame(vi.run())

    return results, vi


def run_policy_iteration(P, R, gamma=0.9, max_iter=1e4):
    pi = PolicyIteration(P, R, gamma, max_iter=max_iter, eval_type="iterative")
    results = pd.DataFrame(pi.run())

    return results, pi

def run_qlearning(P, R, gamma=0.9, alpha=0.1, 
                  alpha_decay=0.99,epsilon=1,
                  epsilon_decay=.99, epsilon_min=0.1, alpha_min=0.01,
                  n_iter=100000, terminating_rewards = [], starting_state=None, n_ep=1000):
    ql = QLearningAdjusted(P, R, gamma, alpha, alpha_decay=alpha_decay,epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, n_iter=n_iter, alpha_min=alpha_min)
    ql.set_terminating_rewards(terminating_rewards)
    ql.set_starting_state(starting_state)

    mean_rewards = []
    threshcount = 0
    last_mean_reward = -1e99
    results = pd.DataFrame(ql.run())
    mean_reward = ql.v_mean[0][-1]
    mean_rewards.append(mean_reward)
    last_policy = ql.policy
    for i in range(n_ep):
        last_mean_reward = mean_reward
        results = ql.run()
        mean_reward = np.mean(ql.v_mean[0])
        mean_rewards.append(mean_reward)
        if i % 100 == 0:
            print(mean_reward, ql.epsilon, ql.alpha, i, sep=" | ")

    return pd.DataFrame(results), ql, mean_rewards

class QLearningAdjusted(QLearning):

    def set_terminating_rewards(self, terminating_rewards):
        self.terminating_rewards = terminating_rewards

    def set_starting_state(self, starting_state):
        self.starting_state = starting_state

    def run(self):

        # Run the Q-learning algorithm.
        error_cumulative = []
        self.run_stats = []
        self.error_mean = []

        v_cumulative = []
        self.v_mean = []

        self.p_cumulative = []

        self.time = time.time()

        # initial state choice
        if self.starting_state is not None:
            s = self.starting_state
        else:
            s = np.random.randint(0, self.S)
        run_stats = []
        for n in range(1, self.max_iter + 1):

            take_run_stat = True# n % self.run_stat_frequency == 0 or n == self.max_iter


            # Action choice : greedy with increasing probability
            # The agent takes random actions for probability ε and greedy action for probability (1-ε).
            pn = np.random.random()
            if pn < self.epsilon:
                a = np.random.randint(0, self.A)
            else:
                # optimal_action = self.Q[s, :].max()
                a = self.Q[s, :].argmax()

            # Simulating next state s_new and reward associated to <s,s_new,a>
            p_s_new = np.random.random()
            p = 0
            s_new = -1
            while (p < p_s_new) and (s_new < (self.S - 1)):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]

            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            # Q[s, a] = Q[s, a] + alpha*(R + gamma*Max[Q(s’, A)] - Q[s, a])
            # Updating the value of Q
            dQ = self.alpha * (r + self.gamma * self.Q[s_new, :].max() - self.Q[s, a])
            self.Q[s, a] = self.Q[s, a] + dQ

            # Computing means all over maximal Q variations values
            error = np.absolute(dQ)

            # compute the value function and the policy
            v = self.Q.max(axis=1)
            self.V = v
            p = self.Q.argmax(axis=1)
            self.policy = p

            self.S_freq[s,a] += 1
            run_stats.append(self._build_run_stat(i=n, s=s, a=a, r=r, p=p, v=v, error=error))

            if take_run_stat:
                error_cumulative.append(error)

                if len(error_cumulative) == 100:
                    self.error_mean.append(np.mean(error_cumulative))
                    error_cumulative = []

                v_cumulative.append(v)

                if len(v_cumulative) == 100:
                    self.v_mean.append(np.mean(v_cumulative, axis=1))
                    v_cumulative = []

                if len(self.p_cumulative) == 0 or not np.array_equal(self.policy, self.p_cumulative[-1][1]):
                    self.p_cumulative.append((n, self.policy.copy()))
                """
                Rewards,errors time at each iteration I think
                But thats for all of them and steps per episode?
                Alpha decay and min ?
                And alpha and epsilon at each iteration?
                """
                self.run_stats.append(run_stats[-1])
                run_stats = []

            if self.iter_callback is not None:
                reset_s = self.iter_callback(s, a, s_new)

            # current state is updated
            s = s_new

            self.alpha *= self.alpha_decay
            if self.alpha < self.alpha_min:
                self.alpha = self.alpha_min

            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

            if r in self.terminating_rewards:
                if not take_run_stat:
                    error_cumulative.append(error)

                    if len(error_cumulative) == 100:
                        self.error_mean.append(np.mean(error_cumulative))
                        error_cumulative = []

                    v_cumulative.append(v)

                    if len(v_cumulative) == 100:
                        self.v_mean.append(np.mean(v_cumulative, axis=1))
                        v_cumulative = []

                    if len(self.p_cumulative) == 0 or not np.array_equal(self.policy, self.p_cumulative[-1][1]):
                        self.p_cumulative.append((n, self.policy.copy()))
                    self.run_stats.append(run_stats[-1])
                    run_stats = []
                break

        self._endRun()
        # add stragglers
        if len(v_cumulative) > 0:
            self.v_mean.append(np.mean(v_cumulative, axis=1))
        if len(error_cumulative) > 0:
            self.error_mean.append(np.mean(error_cumulative))
        if self.run_stats is None or len(self.run_stats) == 0:
            self.run_stats = run_stats
        return self.run_stats