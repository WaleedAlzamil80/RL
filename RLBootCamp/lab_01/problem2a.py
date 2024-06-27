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

expected_val = np.array([  1.381e-18,   1.844e-04,   1.941e-03,   1.272e-03,   2.108e-18,
         0.000e+00,   8.319e-03,   1.727e-16,   3.944e-18,   2.768e-01,
         8.562e-02,  -7.242e-16,   7.857e-18,   3.535e-01,   8.930e-01,
         0.000e+00])

actual_val = compute_vpi(np.arange(16) % mdp.nA, mdp, gamma=GAMMA)
if np.all(np.isclose(actual_val, expected_val, atol=1e-4)):
    print("Test passed")
else:
    print("Expected: ", expected_val)
    print("Actual: ", actual_val)