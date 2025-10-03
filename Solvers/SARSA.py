# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

from collections import defaultdict
import numpy as np
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting

# ==========================================================================================
# AI-GENERATED CODE ATTRIBUTION
# The implementation and revisions to the `Sarsa` class methods
# (`train_episode`, `create_greedy_policy`, and `epsilon_greedy_action`)
# were completed with the assistance of an AI tool (ChatGPT, model “GPT-5”).
# In accordance with course policy, the prompt and the significant parts of the AI's
# responses are documented below.
#
# --- PROMPTS (UNABRIDGED & RELEVANT) ---
# 1) “Complete the three methods, 'train_episode', 'create_greedy_policy', and 
#     'make_epsilon_greedy_policy', within the 'Sarsa' class in 'Sarsa.py' that implements 
#     on-policy TD control (SARSA).”
# 2) “Good, it works. Now comment the code based on this pseudo code. I don’t really know 
#     Python but I can understand the pseudo code. Use the pseudo code notations.”
# 3) “Add comments to the beginning of the code to comply with this [AI attribution policy].”
#
# --- AI RESPONSE CONTENT (ESSENTIALS) ---
# - Implemented `train_episode` according to the SARSA algorithm pseudocode:
#   Initialize S, choose A with ε-greedy policy, loop until terminal:
#   take action A, observe R and S’, choose A’, update Q(S,A),
#   then set S ← S’, A ← A’.
# - Implemented `create_greedy_policy`: returns argmax_a Q(s,a).
# - Implemented `epsilon_greedy_action`: returns ε-greedy probability distribution π(a|s).
# - Added inline comments mapping each Python statement to the pseudocode step.
#
# ==========================================================================================



class Sarsa(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete state spaces"
        )
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def train_episode(self):
        """
        Run one episode of the SARSA algorithm: On-policy TD control.

        Use:
            self.env: OpenAI environment.
            self.epsilon_greedy_action(state): returns an epsilon greedy action
            self.options.steps: number of steps per episode
            self.options.gamma: Gamma discount factor.
            self.options.alpha: TD learning rate.
            self.Q[state][action]: q value for ('state', 'action')
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.

        """

        # Reset the environment, aka Initialize S
        state, _ = self.env.reset()
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

        # Choose A from S using policy derived from Q (ε-greedy)
        action_probs = self.epsilon_greedy_action(state)
        action = np.random.choice(self.env.action_space.n, p=action_probs)

        for _ in range(self.options.steps):
            # Take action A, observe R, S'
            next_state, reward, done, _ = self.step(action)

            # Choose A' from S' using policy derived from Q (ε-greedy)
            next_action_probs = self.epsilon_greedy_action(next_state)
            next_action = np.random.choice(self.env.action_space.n, p=next_action_probs)

            # Update Q(S,A) ← Q(S,A) + α [ R + γ Q(S',A') − Q(S,A) ]
            if done:
                td_target = reward
            else:
                td_target = reward + self.options.gamma * self.Q[next_state][next_action]
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.options.alpha * td_error

            # S ← S'; A ← A'
            state = next_state
            action = next_action

            # until S is terminal
            if done:
                break

    def __str__(self):
        return "Sarsa"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes a state as input and returns a greedy action.
        """

        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes a state as input and returns a greedy action.
        """
        def policy_fn(state):
            # Return argmax_a Q(s,a)
            return int(np.argmax(self.Q[state]))
        return policy_fn

    def epsilon_greedy_action(self, state):
        """
        Return an epsilon-greedy action based on the current Q-values and
        epsilon.

        Use:
            self.env.action_space.n: the size of the action space
            np.argmax(self.Q[state]): action with highest q value
        Returns:
            Probability of taking actions as a vector where each entry is the probability of taking that action
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        """
        Return ε-greedy policy π(a|s) over actions.
        """
        nA = self.env.action_space.n
        eps = self.options.epsilon

        # Initialize all actions with probability ε / |A|
        probs = np.ones(nA, dtype=float) * (eps / nA)

        # Best action = argmax_a Q(s,a)
        best_action = int(np.argmax(self.Q[state]))

        # Assign probability (1-ε) + ε/|A| to greedy action
        probs[best_action] += (1.0 - eps)

        # Return π(a|s) as probability distribution
        return probs

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
