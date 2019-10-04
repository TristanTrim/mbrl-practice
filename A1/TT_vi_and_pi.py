
### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

        P: nested dictionary
                From gym.core.Environment
                For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
                *LIST OF* tuples of the form (probability, nextstate, reward, terminal) where
                        0 - probability: float
                                the probability of transitioning from "state" to "nextstate" with "action"
                        1 - nextstate: int
                                denotes the state we transition to (in range [0, nS - 1])
                        2 - reward: int
                                either 0 or 1, the reward for transitioning from "state" to
                                "nextstate" with "action"
                        3 - terminal: bool
                          True when "nextstate" is a terminal state (hole or goal), False otherwise
        nS: int
                number of states in the environment
        nA: int
                number of actions in the environment
        gamma: float
                Discount factor. Number in range [0, 1)
"""

        #################################
        # YOUR IMPLEMENTATION ALSO HERE #

def print_square(square):
    """For debugging. Prints a 4x4 square nicely."""
    print("_______________________")
    for x in range(0,16,4):
        print(("{:6.2f}"*4).format(*square[x:x+4]))
    print("```````````````````````")

def value_return(P, state, action, value_function, gamma=0.9):
        """ Given a mdp P, a state, an action and a value function, calc the return"""
        outcomes = P[state][action]
        new_value = 0.0
        for outcome in outcomes:
            probability, nextstate, reward, terminal = outcome
            # we are getting rewards based on outcome, not on action, so the reward gets multiplied by the chance of outcome
            new_value += probability*( reward + gamma*value_function[nextstate] )
        return new_value

def policy_evaluation_step(P, nS, nA, policy, value_function, gamma=0.9, tol=1e-3):
        """Perform a single step of policy evaluation on all states. Used in policy_evaluation and value_iteration.

        !!! Makes modifications to value_function !!!
        
        Parameters
        ----------
        All the same things
        
        Returns
        -------
        converged: bool
                True if all the values have converged to within the given tolerance.
        """
        converged = False
        all_states_same = True
        for state in range(nS):
            ###old_value = value_function[state]
            action = policy[state]
            new_value = value_return(P, state, action, value_function, gamma=0.9)
            value_difference = abs(new_value - value_function[state])
            value_function[state] = new_value
            if value_difference > tol:
                all_states_same = False
        if all_states_same:
            converged = True
        return(converged)

        #################################

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
        """Evaluate the value function from a given policy.

        Parameters
        ----------
        P, nS, nA, gamma:
                defined at beginning of file
        policy: np.array[nS]
                The policy to evaluate. Maps states to actions.
        tol: float
                Terminate policy evaluation when
                        max |value_function(s) - prev_value_function(s)| < tol
        Returns
        -------
        value_function: np.ndarray[nS]
                The value function of the given policy, where value_function[s] is
                the value of state s
        """

        value_function = np.zeros(nS)

        ############################
        # YOUR IMPLEMENTATION HERE #
        
        converged = False
        while not converged:
            converged = policy_evaluation_step(P, nS, nA, policy, value_function, gamma, tol)

        ############################
        return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
        """Given the value function from policy improve the policy.

        Parameters
        ----------
        P, nS, nA, gamma:
                defined at beginning of file
        value_from_policy: np.ndarray
                The value calculated from the policy
        policy: np.array
                The previous policy.

        Returns
        -------
        new_policy: np.ndarray[nS]
                An array of integers. Each integer is the optimal action to take
                in that state according to the environment dynamics and the
                given value function.
        """

        new_policy = np.zeros(nS, dtype='int')

        ############################
        # YOUR IMPLEMENTATION HERE #

        for state in range(nS):
            best_value = value_return(P, state, 0, value_from_policy, gamma=0.9)
            for action in range(1,nA):
                action_value = value_return(P, state, action, value_from_policy, gamma=0.9)
                if action_value > best_value:
                    best_value = action_value
                    new_policy[state] = action

        ############################
        return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
        """Runs policy iteration.

        You should call the policy_evaluation() and policy_improvement() methods to
        implement this method.

        Parameters
        ----------
        P, nS, nA, gamma:
                defined at beginning of file
        tol: float
                tol parameter used in policy_evaluation()
        Returns:
        ----------
        value_function: np.ndarray[nS]
        policy: np.ndarray[nS]
        """

        value_function = np.zeros(nS)
        policy = np.zeros(nS, dtype=int)

        ############################
        # YOUR IMPLEMENTATION HERE #

        old_policy_hash = hash(tuple(policy))
        while(True):
            value_function = policy_evaluation(P, nS, nA, policy)
            policy = policy_improvement(P, nS, nA, value_function, policy)
            new_policy_hash = hash(tuple(policy))
            if old_policy_hash == new_policy_hash:
                break
            old_policy_hash = new_policy_hash

        ############################
        return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
        """
        Learn value function and policy by using value iteration method for a given
        gamma and environment.

        Parameters:
        ----------
        P, nS, nA, gamma:
                defined at beginning of file
        tol: float
                Terminate value iteration when
                        max |value_function(s) - prev_value_function(s)| < tol
        Returns:
        ----------
        value_function: np.ndarray[nS]
        policy: np.ndarray[nS]
        """

        value_function = np.zeros(nS)
        policy = np.zeros(nS, dtype=int)
        ############################
        # YOUR IMPLEMENTATION HERE #

        converged = False
        while not converged:
            policy = policy_improvement(P, nS, nA, value_function, policy, gamma=0.9)
            converged = policy_evaluation_step(P, nS, nA, policy, value_function, gamma, tol)

        ############################
        return value_function, policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
          print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

        # comment/uncomment these lines to switch between deterministic/stochastic environments
        #env = gym.make("Deterministic-4x4-FrozenLake-v0")
        env = gym.make("Stochastic-4x4-FrozenLake-v0")

        print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

        V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
        render_single(env, p_pi, 100)

        print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

        V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
        render_single(env, p_vi, 100)


