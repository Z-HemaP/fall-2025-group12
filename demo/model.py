#%%
import numpy as np
import pandas as pd

class NonStationaryEpsilonGreedy:
    """
    Implementation of the Epsilon-Greedy algorithm for nonstationary multi-armed bandits,
    using exponential-recent-weight update matching the formula:
    Q_{n+1} = (1-alpha)^n Q1 + sum_{i=1}^n alpha*(1-alpha)^{n-i}*R_i
    """
    def __init__(self, n_arms, epsilon=0.1, alpha=0.1, manual_prior=None, random_seed=None):
        np.random.seed(random_seed)
        self.n_arms = n_arms
        if manual_prior:
            self.Q = np.ones((n_arms, 1)) * manual_prior[0]
            self.N = np.ones((n_arms, 1)) * manual_prior[1]
        else:
            self.Q = np.zeros((n_arms, 1))
            self.N = np.zeros((n_arms, 1))
        self.epsilon = epsilon
        self.alpha = alpha
        self.arm_rewards = [[] for _ in range(n_arms)]  # arm-wise reward histories
        self.rewards_list = []  # global reward history for plotting
        self.rewards_matrix = np.zeros((1, n_arms))

    def bandit(self, data, A, t):
        reward = data[t][A]
        self.arm_rewards[A].append(reward)
        self.rewards_list.append(reward)
        self.rewards_matrix = np.vstack([self.rewards_matrix, data[t].reshape((1, -1))])
        return reward

    def action(self):
        if np.random.uniform(0, 1) <= self.epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            return np.argmax(self.Q)

    def update(self, A):
        rewards = self.arm_rewards[A]
        n = len(rewards)
        Q1 = self.Q[A][0] 
        if n == 0:
            return
        Q_new = (1 - self.alpha) ** n * Q1
        
        Q_new += np.sum([self.alpha * (1 - self.alpha) ** (n - i) * rewards[i-1] for i in range(1, n+1)])
        self.Q[A] = Q_new

    def train(self, data):
        for t in range(len(data)):
            A = self.action()
            R = self.bandit(data, A, t)
            self.update(A)
            self.N[A] += 1
        table = self.create_table()
        return table, self.rewards_list, self.rewards_matrix[1:, :]

    def create_table(self):
        table = np.hstack([
            np.arange(1, self.n_arms + 1).reshape(self.n_arms, 1),
            self.N,
            self.Q.round(2)
        ]).astype(float)
        table = pd.DataFrame(data=table, columns=["Arms", "Arm Selection", "E(reward|action)"])
        return table.to_string(index=False)

    def save(self):
        df = pd.DataFrame({"N": self.N.flatten(), 'Q': self.Q.flatten()})
        return df

    def reset(self):
        self.Q = np.zeros((self.n_arms, 1))
        self.N = np.zeros((self.n_arms, 1))
        self.arm_rewards = [[] for _ in range(self.n_arms)]
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, self.n_arms))
# --------------------------
# UCB
# --------------------------

class UCB:
    def __init__(self, n_arms, manual_prior=None, random_seed=None):
        np.random.seed(random_seed)
        self.n_arms = n_arms
        if manual_prior:
            self.Q = np.ones((n_arms, 1)) * manual_prior[0]
            self.N = np.ones((n_arms, 1)) * manual_prior[1]
        else:
            self.Q = np.zeros((n_arms, 1))
            self.N = np.zeros((n_arms, 1))
        self.T = 0
        self._reset_logs()

    def _reset_logs(self):
        self.rewards_list = []
        self.chosen_arms = []
        self.rewards_matrix = np.zeros((0, self.n_arms))
        self.cumulative_regret = []

    def bandit(self, data, A, t):
        r_vec = data[t].copy()
        R = int(np.round(r_vec[A]))
        row = np.zeros_like(r_vec); row[A] = R
        self.rewards_matrix = np.vstack([self.rewards_matrix, row.reshape(1, -1)])
        self.rewards_list.append(R)
        return R, r_vec
    

    def action(self):
        self.T += 1
        # Pull each arm once
        for a in range(self.n_arms):
            if self.N[a] == 0:
                return a
        bonus = np.sqrt((2.0 * np.log(self.T)) / self.N.flatten())
        ucb = self.Q.flatten() + bonus
        return int(np.argmax(ucb))


    def update(self, A, R):
        self.N[A] += 1
        self.Q[A] += (1.0 / self.N[A]) * (R - self.Q[A])

    def train(self, data, means_t=None):
        self._reset_logs()
        self.T = 0
        cum_regret = 0.0

        T = data.shape[0]
        for t in range(T):
            A = self.action()
            self.chosen_arms.append(A)

            R, r_vec = self.bandit(data, A, t)
            self.update(A, R)

            regret_t = float(np.max(r_vec) - R)
            cum_regret += regret_t
            self.cumulative_regret.append(cum_regret)

        table = self.create_table()
        return table, self.rewards_list, self.rewards_matrix #, self.cumulative_regret

    def create_table(self):
        table = np.hstack([
            np.arange(1, self.n_arms + 1).reshape(self.n_arms, 1),
            self.N,
            self.Q.round(3)
        ]).astype(float)
        df = pd.DataFrame(table, columns=["Arm", "Count", "E[reward|arm]"])
        return df.to_string(index=False)

    def save(self):
        return pd.DataFrame({"N": self.N.flatten(), "Q": self.Q.flatten()})

    def recommend(self):
        best = int(np.argmax(self.Q))
        return best, float(self.Q[best])

    def reset(self):
        self.Q[:] = 0.0
        self.N[:] = 0.0
        self.T = 0
        self._reset_logs()


# --------------------------
# SLIDING-WINDOW UCB
# --------------------------

import numpy as np
import pandas as pd
from collections import deque

class SlidingWindowUCB:
    """
    Sliding-Window UCB (SW-UCB) for nonstationary multi-armed bandits.
    Uses only rewards from the last `tau` rounds for each arm.
    """

    def __init__(self, n_arms, window, B=1.0, xi=1.0, manual_prior=None, random_seed=None):
        np.random.seed(random_seed)
        self.n_arms = n_arms
        self.tau = window          # window length
        self.B = B              # reward bound
        self.xi = xi            # exploration parameter (> 1/2 in theory)

        # Per-arm sliding windows of (t, reward)
        self.windows = [deque() for _ in range(n_arms)]

        if manual_prior:
            self.Q = np.ones((n_arms, 1)) * manual_prior[0]
            self.N = np.ones((n_arms, 1)) * manual_prior[1]
        else:
            self.Q = np.zeros((n_arms, 1))   # sliding-window mean per arm
            self.N = np.zeros((n_arms, 1))   # total pulls per arm

        self.arm_rewards = [[] for _ in range(n_arms)]
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, n_arms))
        self.t = 0  # global time step (1-based in formulas)

    def bandit(self, data, A, t):
        reward = data[t][A]
        self.arm_rewards[A].append(reward)
        self.rewards_list.append(reward)
        self.rewards_matrix = np.vstack(
            [self.rewards_matrix, data[t].reshape((1, -1))]
        )
        return reward

    def _ucb_scores(self):
        scores = np.zeros(self.n_arms)
        log_term = np.log(max(1, min(self.t, self.tau)))
        for i in range(self.n_arms):
            n_i = len(self.windows[i])
            if n_i == 0:
                scores[i] = float("inf")  # force exploration
            else:
                mean_i = self.Q[i][0]
                bonus = self.B * np.sqrt(self.xi * log_term / n_i)
                scores[i] = mean_i + bonus
        return scores

    def action(self):
        # Initialization phase: play each arm once
        if self.t <= self.n_arms:
            return self.t - 1  # arm index (0-based)
        
        # UCB selection after initialization
        scores = self._ucb_scores()
        return int(np.argmax(scores))

    def update(self, A):
        # newest reward is the last one for arm A
        if not self.arm_rewards[A]:
            return
        reward = self.arm_rewards[A][-1]

        # add (time, reward) to window
        self.windows[A].append((self.t, reward))

        # remove rewards older than t - tau
        while self.windows[A] and self.windows[A][0][0] <= self.t - self.tau:
            self.windows[A].popleft()

        # recompute sliding-window mean
        rewards = [r for (_, r) in self.windows[A]]
        self.Q[A] = np.mean(rewards) if rewards else 0.0

    def train(self, data):
        T = len(data)
        for t in range(T):
            self.t = t + 1
            A = self.action()
            R = self.bandit(data, A, t)
            self.update(A)
            self.N[A] += 1
        table = self.create_table()
        return table, self.rewards_list, self.rewards_matrix[1:, :]

    def create_table(self):
        table = np.hstack([
            np.arange(1, self.n_arms + 1).reshape(self.n_arms, 1),
            self.N,
            self.Q.round(2)
        ]).astype(float)
        table = pd.DataFrame(
            data=table,
            columns=["Arms", "Arm Selection", "E(reward|action)"]
        )
        return table.to_string(index=False)

    def save(self):
        df = pd.DataFrame({"N": self.N.flatten(), "Q": self.Q.flatten()})
        return df

    def reset(self):
        self.windows = [deque() for _ in range(self.n_arms)]
        self.Q = np.zeros((self.n_arms, 1))
        self.N = np.zeros((self.n_arms, 1))
        self.arm_rewards = [[] for _ in range(self.n_arms)]
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, self.n_arms))
        self.t = 0


# class SlidingWindowUCB:
#     """
#     Sliding-Window UCB (SW-UCB) implementation for nonstationary bandit problems,
#     as introduced in Garivier & Moulines (2008).
#     Uses only the τ most recent rewards for each arm when computing empirical means.
#     """
#     def __init__(self, n_arms, window=100, random_seed=None):
#         np.random.seed(random_seed)
#         self.n_arms = n_arms
#         self.window = window
#         self.arm_rewards = [[] for _ in range(n_arms)]
#         self.t = 0
#         self._reset_logs()

#     def _reset_logs(self):
#         self.rewards_list = []
#         self.chosen_arms = []
#         self.rewards_matrix = np.zeros((0, self.n_arms))

#     def bandit(self, data, A, t):
#         R = data[t][A]
#         self.rewards_matrix = np.vstack([self.rewards_matrix, data[t].reshape(1, -1)])
#         self.arm_rewards[A].append(R)
#         if len(self.arm_rewards[A]) > self.window:
#             self.arm_rewards[A].pop(0)
#         self.rewards_list.append(R)
#         self.chosen_arms.append(A)
#         return R

#     def action(self):
#         self.t += 1
#         for a in range(self.n_arms):
#             if len(self.arm_rewards[a]) == 0:
#                 return a
#         means = np.array([np.mean(r) if r else 0. for r in self.arm_rewards])
#         counts = np.array([len(r) for r in self.arm_rewards])
#         # bound the window size
#         effective_counts = np.minimum(counts, self.window)
#         bonuses = np.sqrt((2 * np.log(min(self.t, self.window))) / effective_counts)
#         bonuses[effective_counts == 0] = np.inf
#         ucb = means + bonuses
#         return int(np.argmax(ucb))

#     def update(self, A, R):
#         pass  # Update is handled in bandit()

#     def train(self, data):
#         self._reset_logs()
#         self.t = 0
#         for t in range(data.shape[0]):
#             A = self.action()
#             R = self.bandit(data, A, t)
#         table = self.create_table()
#         return table, self.rewards_list, self.rewards_matrix

#     def create_table(self):
#         N = np.array([len(r) for r in self.arm_rewards])
#         Q = np.array([np.mean(r[-self.window:]) if r else 0. for r in self.arm_rewards])
#         table = np.hstack([
#             np.arange(1, self.n_arms + 1).reshape(-1, 1),
#             N.reshape(-1, 1),
#             Q.reshape(-1, 1),
#         ])
#         df = pd.DataFrame(table, columns=["Arm", "Count", "E[reward|arm]"])
#         return df.to_string(index=False)

#     def reset(self):
#         self.__init__(self.n_arms, self.window)




# --------------------------
# CUMSUM-UCB
# --------------------------

import numpy as np
import pandas as pd

class CUSUMUCB:
    """
    CUSUM-UCB for nonstationary multi-armed bandits.

    - UCB1 selection:
        UCB_i(t) = Q[i] + sqrt( (2 * log(t+1)) / N[i] )
      (when N[i] > 0; otherwise force exploration)

    - Per-arm CUSUM change detection:
        g_plus[i]  = max(0, g_plus[i]  + (r - u_hat[i] - epsilon))
        g_minus[i] = max(0, g_minus[i] + (u_hat[i] - r - epsilon))

      If g_plus[i] >= h or g_minus[i] >= h:
          change detected -> reset that arm's stats.
    """

    def __init__(self, n_arms, epsilon=0.05, h=10.0,
                 u_init=0.5, alpha = 0.2, random_seed=None):
        """
        n_arms : number of arms
        epsilon: CUSUM slack parameter (small positive)
        h      : CUSUM threshold
        u_init : initial guess for pre-change mean (scalar, e.g. 0.5 for Bernoulli)
        """
        np.random.seed(random_seed)
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.h = h
        self.alpha = alpha  # exploration probability for epsilon-greedy part

        # Pre-change mean estimate u_hat[i]
        # In simple use, we keep this fixed to u_init; can update it if desired.
        self.u_hat = np.ones((n_arms, 1)) * u_init

        # CUSUM statistics
        self.g_plus = np.zeros((n_arms, 1))
        self.g_minus = np.zeros((n_arms, 1))

        # Empirical means and counts since last (re)start
        self.Q = np.zeros((n_arms, 1))        # mean reward since last reset
        self.N = np.zeros((n_arms, 1))        # pulls since last reset

        self.arm_rewards = [[] for _ in range(n_arms)]
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, n_arms))

        # Time step
        self.t = 0

    def bandit(self, data, A, t):
        """
        Return reward for pulling arm A at time t, record global histories.
        """
        reward = data[t][A]
        self.arm_rewards[A].append(reward)
        self.rewards_list.append(reward)
        self.rewards_matrix = np.vstack(
            [self.rewards_matrix, data[t].reshape((1, -1))]
        )
        return reward

    def action(self):
    # Increment global time
        self.t += 1

        # Force each arm to be pulled at least once
        for a in range(self.n_arms):
            if self.N[a] == 0:
                return a

        # Decide: exploitation (UCB) vs uniform exploration
        if np.random.rand() <= 1.0 - self.alpha:
            # UCB part: I_t = argmax_i (X̄_t(i) + C_t(i))
            bonus = np.sqrt((2.0 * np.log(self.t)) / self.N.flatten())
            ucb = self.Q.flatten() + bonus
            return int(np.argmax(ucb))
        else:
            # Uniform over all arms: P(i) = α / K
            return np.random.randint(0, self.n_arms)

    def _update_cusum(self, A, reward):
        """
        Update CUSUM for arm A, return True if change detected.
        """
        # Using scalar u_hat for each arm
        mean_ref = self.u_hat[A][0]

        s_plus = reward - mean_ref - self.epsilon
        s_minus = mean_ref - reward - self.epsilon

        self.g_plus[A][0] = max(0.0, self.g_plus[A][0] + s_plus)
        self.g_minus[A][0] = max(0.0, self.g_minus[A][0] + s_minus)

        if self.g_plus[A][0] >= self.h or self.g_minus[A][0] >= self.h:
            return True
        return False

    def _reset_arm(self, A):
        """
        Reset stats for arm A after change detection.
        CUSUM stats go to 0; UCB stats (Q, N) restart.
        """
        self.g_plus[A][0] = 0.0
        self.g_minus[A][0] = 0.0
        self.Q[A][0] = 0.0
        self.N[A][0] = 0.0
        
    def _update_ucb_stats(self, A, reward):
        """
        Incremental mean update for Q[A], and count N[A].
        """
        n_old = self.N[A][0]
        q_old = self.Q[A][0]

        n_new = n_old + 1.0
        q_new = q_old + (reward - q_old) / n_new

        self.N[A][0] = n_new
        self.Q[A][0] = q_new

    def update(self, A, reward):
        """
        Given chosen arm A and obtained reward, update:
        - CUSUM stats for A (and possibly reset if change detected)
        - UCB stats for A (if no change detected)
        """
        change_detected = self._update_cusum(A, reward)

        if change_detected:
            self._reset_arm(A)
        else:
            # No change: update UCB stats
            self._update_ucb_stats(A, reward)

    def train(self, data):
        """
        Run CUSUM-UCB over the full dataset (rows = time, cols = arms).
        """
        T = len(data)
        for t in range(T):
            self.t = t
            A = self.action()
            R = self.bandit(data, A, t)
            self.update(A, R)
        table = self.create_table()
        return table, self.rewards_list, self.rewards_matrix[1:, :]

    def create_table(self):
        """
        Final summary similar to your NonStationaryEpsilonGreedy:
        Arms | Arm Selection Count | E(reward|action)
        """
        table = np.hstack([
            np.arange(1, self.n_arms + 1).reshape(self.n_arms, 1),
            self.N,
            self.Q.round(4)
        ]).astype(float)
        df = pd.DataFrame(
            data=table,
            columns=["Arms", "Arm Selection", "E(reward|action)"]
        )
        return df.to_string(index=False)

    def save(self):
        """
        Save basic stats to a DataFrame.
        """
        df = pd.DataFrame({
            "N": self.N.flatten(),
            "Q": self.Q.flatten(),
            "g_plus": self.g_plus.flatten(),
            "g_minus": self.g_minus.flatten(),
            "u_hat": self.u_hat.flatten()
        })
        return df

    def reset(self):
        """
        Reset everything (full restart of the algorithm).
        """
        self.g_plus = np.zeros((self.n_arms, 1))
        self.g_minus = np.zeros((self.n_arms, 1))
        self.Q = np.zeros((self.n_arms, 1))
        self.N = np.zeros((self.n_arms, 1))
        self.arm_rewards = [[] for _ in range(self.n_arms)]
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, self.n_arms))
        self.t = 0


# class CUSUMUCB:    
#     """
#     CUSUM-UCB from Liu & Lee (2018): resets statistics for each arm when a change is detected via CUSUM.
#     """
#     def __init__(self, n_arms, h=10, epsilon=0.1, M=10, random_seed=None):
#         np.random.seed(random_seed)
#         self.n_arms = n_arms
#         self.h = h
#         self.epsilon = epsilon
#         self.M = M
#         self.N = np.zeros(n_arms)
#         self.Q = np.zeros(n_arms)
#         self.t = 0
#         self.gp = np.zeros(n_arms)  # positive drift
#         self.gn = np.zeros(n_arms)  # negative drift
#         self.last_reset = np.zeros(n_arms, dtype=int)
#         self.arm_rewards = [[] for _ in range(n_arms)]
#         self._reset_logs()

#     def _reset_logs(self):
#         self.rewards_list = []
#         self.chosen_arms = []
#         self.rewards_matrix = np.zeros((0, self.n_arms))

#     def bandit(self, data, A, t):
#         R = data[t][A]
#         self.rewards_matrix = np.vstack([self.rewards_matrix, data[t].reshape(1, -1)])
#         self.rewards_list.append(R)
#         self.chosen_arms.append(A)
#         self.arm_rewards[A].append(R)
#         return R

#     def action(self):
#         self.t += 1
#         for a in range(self.n_arms):
#             if self.N[a] == 0:
#                 return a
#         bonuses = np.sqrt(2 * np.log(max(1, self.t)) / (self.N + 1e-8))  # avoid divide by zero
#         ucb = self.Q + bonuses
#         return int(np.argmax(ucb))

#     def update(self, A, R):
#         self.N[A] += 1
#         self.Q[A] += (R - self.Q[A]) / self.N[A]
#         # CUSUM: Algorithm 2 of paper
#         y = R - self.Q[A]
#         s_plus = y - self.epsilon / 2
#         s_minus = -y - self.epsilon / 2
#         self.gp[A] = max(0, self.gp[A] + s_plus)
#         self.gn[A] = max(0, self.gn[A] + s_minus)
#         if self.gp[A] > self.h or self.gn[A] > self.h:
#             M_eff = min(self.M, len(self.arm_rewards[A]))
#             self.N[A] = M_eff
#             self.Q[A] = np.mean(self.arm_rewards[A][-M_eff:]) if M_eff > 0 else 0.
#             self.gp[A], self.gn[A] = 0, 0
#             self.last_reset[A] = self.t

#     def train(self, data):
#         self._reset_logs()
#         self.t = 0
#         for t in range(data.shape[0]):
#             A = self.action()
#             R = self.bandit(data, A, t)
#             self.update(A, R)
#         table = self.create_table()
#         return table, self.rewards_list, self.rewards_matrix

#     def create_table(self):
#         df = pd.DataFrame({
#             "Arm": np.arange(1, self.n_arms + 1),
#             "Count": self.N,
#             "E[reward|arm]": np.round(self.Q, 3),
#         })
#         return df.to_string(index=False)

#     def reset(self):
#         self.__init__(self.n_arms, self.h, self.epsilon, self.M)

# --------------------------
# EXP3.S
# --------------------------

class EXP3S:
    """
    EXP3.S as in Chen et al (2022): adversarial/stationary bandit with switch-adaptive updates.
    """
    def __init__(self, n_arms, gamma=0.07, alpha=0.02, random_seed=None):
        np.random.seed(random_seed)
        self.n_arms = n_arms
        self.gamma = gamma
        self.alpha = alpha
        self.weights = np.ones(n_arms)
        self.t = 0
        self._reset_logs()

    def _reset_logs(self):
        self.rewards_list = []
        self.chosen_arms = []
        self.rewards_matrix = np.zeros((0, self.n_arms))

    def action(self):
        self.t += 1
        w_sum = np.sum(self.weights)
        probs = (1 - self.gamma) * self.weights / w_sum + self.gamma / self.n_arms
        self.last_probs = probs
        return np.random.choice(self.n_arms, p=probs)

    def bandit(self, data, A, t):
        R = data[t][A]
        self.rewards_matrix = np.vstack([self.rewards_matrix, data[t].reshape(1, -1)])
        self.rewards_list.append(R)
        self.chosen_arms.append(A)
        return R

    def update(self, A, R):
        x_hat = np.zeros(self.n_arms)
        x_hat[A] = R / self.last_probs[A]
        self.weights = self.weights * np.exp(self.gamma * x_hat / self.n_arms)
        # Switch-adaptive step:
        self.weights = (1 - self.alpha) * self.weights + self.alpha * np.mean(self.weights)

    def train(self, data):
        self._reset_logs()
        for t in range(data.shape[0]):
            A = self.action()
            R = self.bandit(data, A, t)
            self.update(A, R)
        table = self.create_table()
        return table, self.rewards_list, self.rewards_matrix

    def create_table(self):
        probs = self.weights / np.sum(self.weights)
        table = np.hstack([
            np.arange(1, self.n_arms + 1).reshape(self.n_arms, 1),
            probs.reshape(-1, 1),
        ])
        df = pd.DataFrame(table, columns=["Arm", "FinalProb"])
        return df.to_string(index=False)

    def reset(self):
        self.__init__(self.n_arms, self.gamma, self.alpha)
#%%
