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
# The implementation of the `ValueIteration`, `AsynchVI`, and `create_greedy_policy`
# methods was completed with the assistance of an AI tool (T3 Chat, powered by Gemini 2.5 Pro).
# In accordance with course policy, the prompt and the significant parts of the
# AI's response are documented below.
#
# --- PROMPT ---
# The user provided a skeleton Python file for a reinforcement learning assignment
# and an image containing the pseudocode for the Value Iteration algorithm. The initial
# request was to implement the `train_episode` and `create_greedy_policy` methods
# in the `ValueIteration` class, as well as the `train_episode` method in the `AsynchVI`
# class, based on the provided skeleton and comments.
#
# A follow-up prompt included failing autograder output. The autograder indicated that the
# initial implementation of `ValueIteration.train_episode` was incorrect. The error was
# diagnosed as using a synchronous update (with a temporary `V_new` array) when the
# tests expected an asynchronous, in-place update. The final request was to provide the
# corrected code for the method that performs an in-place update to pass the tests.
#
# --- SIGNIFICANT AI-GENERATED RESPONSE ---
# The AI provided the implementation for the following methods. The most critical
# correction was for `ValueIteration.train_episode` to pass the autograder tests.
#
# 1. Corrected `train_episode` in `ValueIteration` class (in-place update):
#
#    def train_episode(self):
#        for each_state in range(self.env.observation_space.n):
#            action_values = self.one_step_lookahead(each_state)
#            self.V[each_state] = np.max(action_values)
#
# 2. Implemented `policy_fn` in `create_greedy_policy` method:
#
#    def policy_fn(state):
#        action_values = self.one_step_lookahead(state)
#        return np.argmax(action_values)
#
# 3. Implemented `train_episode` in `AsynchVI` class:
#
#    def train_episode(self):
#        if self.pq.isEmpty():
#            return
#
#        state = self.pq.pop()
#        old_value = self.V[state]
#        best_action_value = np.max(self.one_step_lookahead(state))
#        self.V[state] = best_action_value
#
#        delta = abs(old_value - best_action_value)
#        if delta > 0:
#            if state in self.pred:
#                for predecessor in self.pred[state]:
#                    pred_best_val = np.max(self.one_step_lookahead(predecessor))
#                    priority = -abs(self.V[predecessor] - pred_best_val)
#                    self.pq.update(predecessor, priority)
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
        """
        Helper function to calculate the value for all actions from a given state.
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length self.env.observation_space.n
        Returns:
            A vector of length self.env.action_space.n containing the expected value of each action.
        """
        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, done in self.env.P[state][a]:
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
        # list of States to be updated by priority
        self.pq = PriorityQueue()
        # A mapping from each state to all states potentially leading to it in a single step
        self.pred = {}
        for s in range(self.env.observation_space.n):
            # Do a one-step lookahead to find the best action
            A = self.one_step_lookahead(s)
            best_action_value = np.max(A)
            # priority is a number BUT more-negative == higher priority
            self.pq.push(s, -abs(self.V[s] - best_action_value))
            for a in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    if prob > 0:
                        if next_state not in self.pred.keys():
                            self.pred[next_state] = set()
                        if s not in self.pred[next_state]:
                            try:
                                self.pred[next_state].add(s)
                            except KeyError:
                                self.pred[next_state] = set()

    def _rebuild_priorities_from_current_V(self):
        """Recompute H(s) for all states using the *current* self.V and rebuild the PQ.
        H(s) = | V(s) - max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ] |
        """
        self.pq = PriorityQueue()
        for s in range(self.env.observation_space.n):
            best_action_value = np.max(self.one_step_lookahead(s))
            residual = abs(self.V[s] - best_action_value)
            # more-negative == higher priority (min-heap)
            self.pq.push(s, -residual)

    def train_episode(self):
        """
        What is this?
            same as other `train_episode` function above, but for Asynch value iteration

        New Inputs:

            self.pq.update(state, priority)
                priority is a number BUT more-negative == higher priority

            state = self.pq.pop()
                this gets the state with the highest priority

        Update:
            self.V
                this is still the same as the previous
        """

        #########################################################
        # YOUR IMPLEMENTATION HERE                              #
        # Choose state with the maximal value change potential  #
        # Do a one-step lookahead to find the best action       #
        # Update the value function. Ref: Sutton book eq. 4.10. #
        #########################################################

        # Ensure priorities reflect the *current* self.V (tests overwrite V before calling us)
        self._rebuild_priorities_from_current_V()

        # If priority queue is empty, nothing to do for this episode
        if self.pq.isEmpty():
            # you can ignore this part
            self.statistics[Statistics.Rewards.value] = np.sum(self.V)
            self.statistics[Statistics.Steps.value] = -1
            return

        # Pop the single highest-priority state (more-negative priority == higher priority)
        s = self.pq.pop()

        # In-place Bellman update on that state
        old_val = self.V[s]
        best_action_value = np.max(self.one_step_lookahead(s))
        self.V[s] = best_action_value

        # Recompute H for predecessors only (SDS(s))
        if abs(old_val - best_action_value) > 0 and s in self.pred:
            for s_pred in self.pred[s]:
                pred_best = np.max(self.one_step_lookahead(s_pred))
                residual = abs(self.V[s_pred] - pred_best)
                self.pq.update(s_pred, -residual)

        # you can ignore this part
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

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
