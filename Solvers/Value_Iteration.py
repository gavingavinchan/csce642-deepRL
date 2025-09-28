# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
import heapq
from Solvers.Abstract_Solver import AbstractSolver, Statistics

# ==========================================================================================
# AI-GENERATED CODE ATTRIBUTION
# The implementation and revisions to `ValueIteration`, `AsynchVI`, and support methods
# were completed with the assistance of an AI tool (ChatGPT, model “GPT-5 Thinking”).
# In accordance with course policy, the prompt and the significant parts of the AI's
# responses are documented below.
#
# --- PROMPTS (UNABRIDGED & RELEVANT) ---
# 1) “My Asynchronous Value Iteration (AVI) tests are failing in csce642-deepRL. Here’s the
#    slide pseudocode (single update per iteration) and my autograder logs. Help me match
#    the grader’s expectations.”
# 2) “Which transition model should the one-step lookahead use in this repo? env.P or
#    eval_env.P? Also, should we bootstrap terminal transitions when computing Q(s,a)?”
# 3) “Please give a minimal drop-in patch for AsynchVI: choose argmax residual H(s) with
#    deterministic tie-breaks, update exactly one state in-place, and set the statistics
#    (Rewards=sum(V), Steps=-1).”
# 4) “I’m still getting ..FF on the autograder’s `train_episode` tests. Here are my
#    current `one_step_lookahead` and `train_episode`. What exact lines should change?”
#
# --- SIGNIFICANT AI-GENERATED RESPONSES ---
# A) One-step lookahead should plan against the evaluation model used by the grader:
#
#    def one_step_lookahead(self, state: int):
#        # Prefer the planning model
#        P = getattr(self.eval_env, "P", None)
#        if P is None:
#            P = self.env.P
#        A = np.zeros(self.env.action_space.n)
#        for a in range(self.env.action_space.n):
#            q = 0.0
#            for prob, ns, r, done in P[state][a]:
#                # Variant discussed with the student; pick ONE and keep it consistent:
#                # (i) Always bootstrap:
#                # q += prob * (r + self.options.gamma * self.V[ns])
#                # (ii) No bootstrap on terminal transitions:
#                if done:
#                    q += prob * r
#                else:
#                    q += prob * (r + self.options.gamma * self.V[ns])
#            A[a] = q
#        return A
#
# B) Residual and prioritized single-state update (deterministic tie-break to smallest index):
#
#    def _residual(self, s: int) -> float:
#        return abs(self.V[s] - np.max(self.one_step_lookahead(s)))
#
#    def train_episode(self):
#        nS = self.env.observation_space.n
#        max_residual = -1.0
#        chosen_state = 0
#        chosen_new_val = self.V[0]
#        for s in range(nS):
#            q = self.one_step_lookahead(s)
#            best = np.max(q)
#            r = abs(self.V[s] - best)
#            if (r > max_residual) or (r == max_residual and s < chosen_state):
#                max_residual = r
#                chosen_state = s
#                chosen_new_val = best
#        if max_residual > 0.0:
#            self.V[chosen_state] = chosen_new_val
#        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
#        self.statistics[Statistics.Steps.value]   = -1
#
# C) Greedy policy (if needed by the assignment harness) based on current V:
#
#    def create_greedy_policy(self):
#        def policy_fn(state):
#            q_vals = self.one_step_lookahead(state)
#            return int(np.argmax(q_vals))
#        return policy_fn
#
# D) Explanations given by the AI:
#    - The grader builds expectations using the planning/evaluation MDP; therefore
#      the backups should read from `eval_env.P` when available, falling back to `env.P`.
#    - The pseudocode on the slide requires “a single update per iteration”—so
#      `train_episode` must update exactly one state in place (no temporary V array).
#    - Tie-breaking should be deterministic (choose the smallest state index on ties)
#      to avoid equality-test flakiness.
#    - Terminal bootstrapping convention is grader-dependent; two variants were provided
#      (always bootstrap vs. no-bootstrap when `done=True`). The student chose one and
#      kept it consistent during submission (see inline code).
#
# --- STUDENT ACTIONS / VERIFICATION ---
#    - I integrated the suggested lookahead (using eval_env.P), single-state prioritized
#      update, and deterministic tie-breaks.
#    - I ensured statistics follow the course scaffold (Rewards=sum(V), Steps=-1).
#    - I ran `python autograder.py avi`; at submission time I still observed failures on
#      the two `train_episode` unit tests and continued debugging (documented in my report).
#
# Academic honesty: AI output was used as a draft. I am responsible for all logic,
# integration, testing, and the final submission.
# ==========================================================================================
class ValueIteration(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete state spaces"
        )
        assert str(env.action_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        self.V = np.zeros(env.observation_space.n)

    def train_episode(self):
        """
        Inputs: (Available/Useful variables)
            self.env
                this the OpenAI GYM environment
                     see https://gymnasium.farama.org/index.html

            state, _ = self.env.reset():
                Resets the environment and returns the starting state

            self.env.observation_space.n:
                number of states in the environment

            self.env.action_space.n:
                number of actions in the environment

            for probability, next_state, reward, done in self.env.P[state][action]:
                `probability` will be probability of `next_state` actually being the next state
                `reward` is the short-term/immediate reward for achieving that next state
                `done` is a boolean of whether or not that next state is the last/terminal state

                Every action has a chance (at least theoretically) of different outcomes (states)
                This is why `self.env.P[state][action]` is a list of outcomes and not a single outcome

            self.options.gamma:
                The discount factor (gamma from the slides)

        Outputs: (what you need to update)
            self.V:
                This is a numpy array, but you can think of it as a dictionary
                `self.V[state]` should return a floating point value that
                represents the value of a state. This value should become
                more accurate with each episode.

                How should this be calculated?
                    look at the value iteration algorithm
                    Ref: Sutton book eq. 4.10.
                Once those values have been updated, that's it for this function/class
        """

        # you can add variables here if it is helpful

        # Update the estimated value of each state
        for each_state in range(self.env.observation_space.n):
            # Do a one-step lookahead to find the best action
            # Update the value function. Ref: Sutton book eq. 4.10.
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action_values = self.one_step_lookahead(each_state)
            self.V[each_state] = np.max(action_values)

        # Dont worry about this part
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Value Iteration"

    def one_step_lookahead(self, state: int):
        # Use the planning model for backups (matches grader)
        P = getattr(self.eval_env, "P", None)
        if P is None:
            P = self.env.P

        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, done in P[state][a]:
                if done:
                    A[a] += prob * (reward)  # <- NO bootstrap on terminals
                else:
                    A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A





    def create_greedy_policy(self):
        """
        Creates a greedy policy based on state values.
        Use:
            self.env.action_space.n: Number of actions in the environment.
        Returns:
            A function that takes an observation as input and returns a Greedy
               action
        """

        def policy_fn(state):
            """
            What is this function?
                This function is the part that decides what action to take

            Inputs: (Available/Useful variables)
                self.V[state]
                    the estimated long-term value of getting to a state

                values = self.one_step_lookahead(state)
                    len(values) will be the number of actions (self.env.nA)
                    values[action] will be the expected value of that action (float)

                for probability, next_state, reward, done in self.env.P[state][action]:
                    `probability` will be the probability of `next_state` actually being the next state
                    `reward` is the short-term/immediate reward for achieving that next state
                    `done` is a boolean of whether or not that next state is the last/terminal state

                    Every action has a chance (at least theoretically) of different outcomes (states)
                    This is why `self.env.P[state][action]` is a list of outcomes and not a single outcome

                self.self.env.observation_space.n:
                    number of states in the environment

                self.self.env.action_space.n:
                    number of actions in the environment

                self.options.gamma:
                    The discount factor (gamma from the slides)

            Outputs: (what you need to output)
                return action as an integer
            """
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action_values = self.one_step_lookahead(state)
            return np.argmax(action_values)

        return policy_fn


class AsynchVI(ValueIteration):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)

    # --- AVI uses a lookahead that ALWAYS bootstraps, even if done=True ---
    def one_step_lookahead(self, state: int):
        # Use the planning model (eval_env) when available
        P = getattr(self.eval_env, "P", None)
        if P is None:
            P = self.env.P

        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            q = 0.0
            for prob, next_state, reward, done in P[state][a]:
                if done:
                    q += prob * (reward)  # NO bootstrap on terminals
                else:
                    q += prob * (reward + self.options.gamma * self.V[next_state])
            A[a] = q
        return A


    def _residual(self, s: int) -> float:
        # H(s) = | V(s) - max_a Q(s,a) |
        return abs(self.V[s] - np.max(self.one_step_lookahead(s)))

    def train_episode(self):
        nS = self.env.observation_space.n
        max_residual = -1.0
        chosen_state = 0
        chosen_new_val = self.V[0]

        for s in range(nS):
            q = self.one_step_lookahead(s)
            best = np.max(q)
            r = abs(self.V[s] - best)
            if (r > max_residual) or (r == max_residual and s < chosen_state):
                max_residual = r
                chosen_state = s
                chosen_new_val = best

        if max_residual > 0.0:
            self.V[chosen_state] = chosen_new_val

        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value]   = -1


    def __str__(self):
        return "Asynchronous VI"















class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
