# basic setup
from misc import FrozenLakeEnv, make_grader
env = FrozenLakeEnv()
import numpy as np, numpy.random as nr, gymnasium as gym
import matplotlib.pyplot as plt


np.set_printoptions(precision=3)
class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)
GAMMA = 0.95

def compute_vpi(pi, mdp, gamma):
    # use pi[state] to access the action that's prescribed by this policy
#     V = np.ones(mdp.nS) # REPLACE THIS LINE WITH YOUR CODE
    A = np.zeros((mdp.nS, mdp.nS))
    b = np.zeros(mdp.nS)
    for s in range(mdp.nS):
        for prob, next_state, reward in mdp.P[s][pi[s]]:
            b[s] += prob * reward
            A[s, next_state] += prob * gamma
    A -= np.eye(mdp.nS)        
    V = -np.linalg.solve(A, b).T
    return V


def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_state, reward in mdp.P[s][a]:
                Qpi[s,a] += prob * (reward + gamma*vpi[next_state])
    return Qpi


def policy_iteration(mdp, gamma, nIt, grade_print=print):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    grade_print("Iteration | # chg actions | V[0]")
    grade_print("----------+---------------+---------")
    for it in range(nIt):        
        # YOUR CODE HERE
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        # you need to compute qpi which is the state-action values for current pi
        pi = qpi.argmax(axis=1)
        grade_print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis

expected_output = """Iteration | # chg actions | V[0]
----------+---------------+---------
   0      |      1        | -0.00000
   1      |      9        | 0.00000
   2      |      2        | 0.39785
   3      |      1        | 0.45546
   4      |      0        | 0.53118
   5      |      0        | 0.53118
   6      |      0        | 0.53118
   7      |      0        | 0.53118
   8      |      0        | 0.53118
   9      |      0        | 0.53118
  10      |      0        | 0.53118
  11      |      0        | 0.53118
  12      |      0        | 0.53118
  13      |      0        | 0.53118
  14      |      0        | 0.53118
  15      |      0        | 0.53118
  16      |      0        | 0.53118
  17      |      0        | 0.53118
  18      |      0        | 0.53118
  19      |      0        | 0.53118"""

Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20, grade_print=make_grader(expected_output))
plt.plot(Vs_PI)
plt.show()