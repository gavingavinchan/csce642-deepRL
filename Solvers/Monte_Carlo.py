# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

from collections import defaultdict, OrderedDict
import numpy as np
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class MonteCarlo(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith("Discrete") or str(
            env.observation_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete state spaces"
        )
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        self.policy = self.make_epsilon_greedy_policy()

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        # Keeps track of sum and count of returns for each state
        # to calculate an average. We could use an array to save all
        # returns (like in the book) but that's memory inefficient.
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)

    def train_episode(self):
        """
        Run a single episode for (first visit) Monte Carlo Control using Epsilon-Greedy policies.
        """
        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state, _ = self.env.reset()
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # Generate one complete episode
        for _ in range(self.options.steps):
            probs = self.policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            # --- FIX: Unpack 4 values, not 5 ---
            next_state, reward, done, _ = self.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # First-visit MC updates (by state-action)
        G = 0.0
        gamma = self.options.gamma
        # set of (state, action) pairs visited in this episode
        visited_sa = set()

        # Iterate backwards through the episode
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            sa_pair = (s_t, a_t)

            # If this is the first time we've seen this (state, action) pair
            # in this episode (iterating backwards), then update Q
            if sa_pair not in visited_sa:
                visited_sa.add(sa_pair)
                self.returns_sum[sa_pair] += G
                self.returns_count[sa_pair] += 1.0
                self.Q[s_t][a_t] = (
                    self.returns_sum[sa_pair] / self.returns_count[sa_pair]
                )

    def __str__(self):
        return "Monte Carlo"

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-estimates and epsilon.
        """
        nA = self.env.action_space.n
        epsilon = self.options.epsilon

        def policy_fn(observation):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(self.Q[observation])
            A[best_action] += 1.0 - epsilon
            return A

        return policy_fn

    def create_greedy_policy(self):
        """
        Creates a greedy (soft) policy based on Q values.
        """

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            return np.argmax(self.Q[state])

        return policy_fn

    def plot(self, stats, smoothing_window, final=False):
        # For plotting: Create value function from action-value function
        # by picking the best action at each state
        V = defaultdict(float)
        for state, actions in self.Q.items():
            action_value = np.max(actions)
            V[state] = action_value
        plotting.plot_value_function(V, title="Final Value Function")


class OffPolicyMC(MonteCarlo):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        self.C = defaultdict(lambda: np.zeros(env.action_space.n))
        self.target_policy = self.create_greedy_policy()
        self.behavior_policy = self.create_random_policy()

    def train_episode(self):
        """
        Run a single episode of Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
        """
        episode = []
        state, _ = self.env.reset()
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # Generate an episode by following the behavior policy
        for _ in range(self.options.steps):
            probs = self.behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            # --- FIX: Unpack 4 values, not 5 ---
            next_state, reward, done, _ = self.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        G = 0.0  # Sum of discounted rewards
        W = 1.0  # Importance sampling ratio
        gamma = self.options.gamma

        # Process the episode in reverse order
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            # Update C and Q using weighted importance sampling
            self.C[state][action] += W
            self.Q[state][action] += (W / self.C[state][action]) * (
                G - self.Q[state][action]
            )

            target_action = self.target_policy(state)

            if action != target_action:
                break

            behavior_prob = self.behavior_policy(state)[action]
            if behavior_prob == 0:  # Avoid division by zero
                break
            W = W / behavior_prob

    def create_random_policy(self):
        """
        Creates a random policy function.
        """
        nA = self.env.action_space.n
        A = np.ones(nA, dtype=float) / nA

        def policy_fn(observation):
            return A

        return policy_fn

    def __str__(self):
        return "MC+IS"