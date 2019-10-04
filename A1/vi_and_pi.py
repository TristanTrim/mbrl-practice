### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

import pprint
pp = pprint.PrettyPrinter(indent = 4)
import random
import datetime

np.set_printoptions(precision=3)

"""

This code is full of spiders. Don't read it.

It does follow an epsilon-greedy iterative policy update algorithm.

"""

def select_action(choice_values,epsilon,nA):
    if random.random() > epsilon:
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

def apply_return(values,stateActions,rewards,value_function,alpha,gamma):
    S,A = stateActions[0]
    G = value_function[S,A]
    gam = gamma
    for i in range(len(values)-1):
        G = G + \
                alpha*( \
                rewards[i+1] \
                +gam*values[i+1] \
                - values[i] )
        gam*=gam
    value_function[S,A] = G

def print4x4(thing):
    for y in range(4):
        for x in range(4):
            print(thing[4*y+x],end=',\t')
        print()

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

    #return value_function,  [0, 3, 1, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 2, 0]

    #learning rate
    alpha = 0.001
    #random choice factor
    epsilon = 0.1
    # number of values to use in the return
    nBack = 3
    gamma = .95

    for nBack in range(2,6):
        print("\nnBack = {}".format(nBack))
        print("epsilon = {}".format(epsilon))
        print("alpha = {}".format(alpha))
        print("gamma = {}".format(gamma))
        print()
        for step in range(30):
            for S in range(20000):
                S = 0
                values = []
                rewards = [0]
                stateActions = []
                # the episode!
                for i in range(50):
                    A = select_action(value_function[S],epsilon,nA)

                    values.append(value_function[S,A])
                    stateActions.append((S,A))
                    probability,S,reward,terminal = takeAction(P,S,A)
                    rewards.append(reward)

                    if len(values) == nBack:
                        apply_return(values,stateActions,rewards,value_function,alpha,gamma)
                        del values[0]
                        del rewards[0]
                        del stateActions[0]

                    if terminal:
                        break;

                values.append(value_function[S,A])
                stateActions.append((S,A))
                probability,S,reward,terminal = takeAction(P,S,A)
                rewards.append(reward)
                while(len(values)>0):
                    apply_return(values,stateActions,rewards,value_function,alpha,gamma)
                    del values[0]
                    del rewards[0]
                    del stateActions[0]
            if env:
                #show convergence kinda
                for S in range(nS):
                    policy[S] = select_BEST_action(value_function[S])
              #  print("value function:")
              #  print(value_function)
              #  vf = tuple(round(sum(vf),3) for vf in value_function)
              #  print4x4(vf)
              #  print("policy:")
              #  print4x4(policy)
                render_many(env, policy, 100)
                #render_single(env, policy, 100)
                #print(datetime.datetime.now())


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
  wins = wins/10.0
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
    #render_many(env, p_vi, 100)
    #render_single(env, p_vi, 100)


