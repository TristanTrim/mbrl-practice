
#######################################################################################
###                                                                                    #
### This is an initial experiment in working with the gym library.                     #
###                                                                                    #
### I thought the cart pole problem looked fun, but didn't know how to solve it        #
### using reinforcement learning so for my first attempt I used a very basic           #
### implementation of a single hidden layer neural net trained using an                #
### evolutionary process with random mutations.                                        #
###                                                                                    #
### I got it to work but encountered two surprising phenomena:                         #
###                                                                                    #
###     1)                                                                             #
###     When generating single hidden layer policies with random data about one        #
###     in 20 does quite well on the problem, and at least one in 100 is likely        #
###     to solve it outright.                                                          #
###                                                                                    #
###     2)                                                                             #
###     After initial generation of random policies applying mutation to generate      #
###     new policies always decreases performance of the policy. Obviously this        #
###     is due to the way I've implemented things. I originally moved all of the       #
###     weights in the nn by a random amount, and thought that might be the problem    #
###     but moving only one of them also failed to produce improvements. I suspect     #
###     the problem lies in the distance of the mutation, and/or the small amount of   #
###     policies and mutations I'm generating; the nature of the problem possibly      #
###     requires coordinated changes in the interpretation of inputs making the        #
###     probability of improvement given random mutations vary small.                  #
###                                                                                    #
### Because of these, I am interested to see how a reinforcement learning approach,    #
### Improving actions rather than policies, could work on this problem.                #
###                                                                                    #
### Right now I think I will next create a new solution for this same problem, this    #
### time using a process of mapping the inputs into a tabular form and then using a    #
### basic policy improvemnet algorithm.                                                #
###                                                                                    #
### I am concerned that this approach will not be granular enough without a very       #
### large table. I can think of a couple of approaches that may improve the            #
### performance of a basic mapping, but am interested to see the performance of the    #
### basic model.                                                                       #
###                                                                                    #
### I also suspect there are many other approaches to the problem of RL with floating  #
### point inputs, and I look forward to reading about them, but I'm also interested    #
### in experimenting with and extending the basic tabular model I'm already familiar   #
### with.                                                                              #
###                                                                                    #
########################################################################################


import random

policies_per_roll = 50

def gen_policy():
    return [[random.random()*2-1 for _ in range(5)] for __ in range(10)]

def choose(policy, observation):
    return 0<sum([policy[x][4]*sum([policy[x][i] * observation[i] for i in range(4)]) for x in range(10)])

mutation = .001
def mate(p1,p2):
    #baby_policy = [[(p1[i][x]+p2[i][x])/2 for x in range(5)] for i in range(10)]
    if (random.randint(0,2) == 2):
        baby_policy = p2.copy()
    elif (random.randint(0,1)==1):
        baby_policy = p1.copy()
    else:
        return(p1)
    baby_policy[random.randint(0,9)][random.randint(0,4)] += random.random()*2*mutation-mutation
    return baby_policy

print("Generating Random Policies")
policies = [[gen_policy(),0] for _ in range(policies_per_roll)]

import gym
env = gym.make("CartPole-v1")
observation = env.reset()

trials = 5

#for generation in range(2,5):
while(True):
    mutation/=10
    highest = 0
    second_highest = 0
    best = policies[0]#first is worst
    second_best = [1]
    print("Evaluating Policies")
    for policy in policies:
        lasted = 0
        for test in range(trials):
            observation = env.reset()
            for _ in range(500):
              lasted+=1
              action = choose(policy[0],observation)#env.action_space.sample() # your agent here (this takes random actions)
              observation, reward, done, info = env.step(action)

              if done:
                break
        policy[1] = lasted / trials
        if lasted > highest:
            second_highest = highest
            second_best = best
            highest = lasted
            best = policy
        elif lasted > second_highest:
            second_highest = lasted
            second_best = policy
    #mating
    #print("\n\nBest policies:\n{}\n{}\nNow making next gen\n".format(best[1],second_best[1]))
    print(int(best[1]),end='')
    for i in range(int(best[1]/5)):
        print("#", end = '')
    print()

    print("Demonstrating selected policy")
    observation = env.reset()
    lasted = 0
    for _ in range(500):
      lasted+=1
      env.render()
      action = choose(best[0],observation)#env.action_space.sample() # your agent here (this takes random actions)
      if action:
          foo = ">---------####>"+"."*int((best[1]-300)/8)
      else:
          foo = "<####---------<"+"."*int((best[1]-300)/8)

      print(["{0:+.2E}".format(x) for x in observation],foo)
      observation, reward, done, info = env.step(action)


    #policies = [[mate(best[0],second_best[0]),0] for _ in range(100)] # genetic algorithm
    print("Generating Random Policies")
    policies = [[gen_policy(),0] for _ in range(policies_per_roll)] # new batch of random policies
    #  if done:
    #    break
env.close()

