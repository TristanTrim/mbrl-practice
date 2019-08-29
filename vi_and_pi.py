### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

import pprint
pp = pprint.PrettyPrinter(indent = 4)
import random

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def select_action(choice_values,wander,nA):
    if random.random() < wander:
        action = -1
        max_act_value = 0
        for potential_action in range(nA):
            if choice_values[potential_action] > max_act_value:
                max_act_value = choice_values[potential_action]
                action = potential_action
        if action > -1:
            return action
    return random.choice(range(nA))

def select_BEST_action(choice_values):
    action = 0
    max_act_value = 0
    for potential_action in range(len(choice_values)):
        if choice_values[potential_action] > max_act_value:
            max_act_value = choice_values[potential_action]
            action = potential_action
    return action


def takeAction(P,S,A):
    fate = random.random()
    for outcome in P[S][A]:
        if fate < outcome[0]:
            return outcome
        else:
            fate-=outcome[0]

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3, env=None):
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

    value_function = np.zeros((nS,nA))
    policy = np.zeros(nS, dtype=int)

    #learning rate
    alpha = 0.01
    #discounting factor
    gamma = 0.9
    #random choice factor
    wander = 0.9

    for step in range(100):
        for S in range(1000):
            S = 0
            A = select_action(value_function[S],wander,nA)
            probability,nextS,reward,terminal = takeAction(P,S,A)
            for i in range(50):
                nextA = select_action(value_function[nextS],wander,nA)
                value_function[S,A] = value_function[S,A] \
                    + alpha*(reward + gamma*value_function[nextS,nextA] \
                                - value_function[S,A] \
                             )
                #print(value_function[S,A])

                if terminal:
                    break;

                S, A = nextS, nextA
                probability,nextS,reward,terminal = takeAction(P,S,A)
        if env:
            #show convergence kinda
            for S in range(nS):
                policy[S] = select_BEST_action(value_function[S])
            print("            policy: {}".format(policy))
            render_many(env, policy, 100)


    for S in range(nS):
        policy[S] = select_BEST_action(value_function[S])

    #pp.pprint(P)
    #print("    value function: {}".format(value_function))
    print("            policy: {}".format(policy))
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

def render_many(env, policy, max_steps=100):
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

  wins = 0
  for ii in range(1000):
      episode_reward = 0
      ob = env.reset()
      for t in range(max_steps):
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
          break
      if not done:
        #print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
        pass
      else:
          if episode_reward:
            wins+=1
  wins = wins/10
  print("{}% win {}".format(wins,"#"*int(wins)),)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

    # comment/uncomment these lines to switch between deterministic/stochastic environments
    #env = gym.make("Deterministic-4x4-FrozenLake-v0")
    #env = gym.make("Deterministic-8x8-FrozenLake-v0")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")

#    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

#    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
#    render_single(env, p_pi, 100)

    #print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3, env=env)
    render_many(env, p_vi, 100)
    #render_single(env, p_vi, 100)


