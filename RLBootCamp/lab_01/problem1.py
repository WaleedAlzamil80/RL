from bee import OptimalPath
from misc import FrozenLakeEnv, make_grader
env = FrozenLakeEnv()
# print(env.__doc__)

import numpy as np, numpy.random as nr, gymnasium as gym
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

# # Seed RNGs so you get the same printouts as me
# env._seed(0); # from gym.spaces import prng; prng.seed(10)
# # Generate the episode
# env.reset()
# for t in range(100):
    # env.render()
    # a = env.action_space.sample()
    # ob, rew, done, _ = env.step(a)
    # if done:
        # break
# assert done
# env.render()

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states 16
        self.nA = nA # number of actions 4
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)

mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)


# print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
# print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
# print(np.arange(16).reshape(4,4))
# print("Action indices [0, 1, 2, 3] correspond to West, South, East and North.")
# print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
# print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
# print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")
# for i in range(4):
    # print("P[5][%i] =" % i, mdp.P[5][i])

# for state, action in mdp.P.items():
    # print(state)
    # for act, transition in action.items():
        # print(transition)
        # for probability, nextstate, reward in transition:
            # print(probability)

expected_output = """Iteration | max|V-Vprev| | # chg actions | V[0]
----------+--------------+---------------+---------
   0      | 0.80000      |  N/A          | 0.000
   1      | 0.64000      |    2          | 0.000
   2      | 0.57600      |    2          | 0.000
   3      | 0.46080      |    2          | 0.000
   4      | 0.36864      |    2          | 0.000
   5      | 0.32768      |    1          | 0.328
   6      | 0.14254      |    0          | 0.452
   7      | 0.13828      |    1          | 0.590
   8      | 0.05512      |    0          | 0.646
   9      | 0.04397      |    1          | 0.690
  10      | 0.02485      |    1          | 0.708
  11      | 0.01802      |    2          | 0.724
  12      | 0.01739      |    0          | 0.733
  13      | 0.01539      |    1          | 0.742
  14      | 0.01392      |    0          | 0.750
  15      | 0.01237      |    1          | 0.758
  16      | 0.01111      |    0          | 0.765
  17      | 0.00992      |    0          | 0.772
  18      | 0.00891      |    0          | 0.779
  19      | 0.00798      |    0          | 0.785"""

Vs_VI, pis_VI = OptimalPath(mdp, grade_print=make_grader(expected_output))

for (V, pi) in zip(Vs_VI[:10], pis_VI[:10]):
    plt.figure(figsize=(3,3))
    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(4)-.5)
    ax.set_yticks(np.arange(4)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
    Pi = pi.reshape(4,4)
    for y in range(4):
        for x in range(4):
            a = Pi[y, x]
            u, v = a2uv[a]
            plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1) 
            plt.text(x, y, str(env.desc[y,x].item().decode()),
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')
    plt.show()
plt.figure()
plt.plot(Vs_VI)
plt.title("Values of different states")
plt.show()