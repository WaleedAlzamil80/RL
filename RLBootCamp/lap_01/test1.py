from misc import FrozenLakeEnv
env = FrozenLakeEnv()
# print(env.__doc__)

import numpy as np, numpy.random as nr, gymnasium as gym
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P  # state transition and reward probabilities
        self.nS = nS  # number of states
        self.nA = nA  # number of actions
        self.desc = desc  # 2D array specifying what each grid cell means (used for plotting)


mdp = MDP(
    {s: {a: [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()},
    env.nS,
    env.nA,
    env.desc
)

def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations
    Outputs:
        (value_functions, policies)
        
    len(value_functions) == nIt+1 and len(policies) == nIt
    """
    Vs = [np.zeros(mdp.nS)]  # list of value functions, starting with V^{(0)} = 0
    pis = []

    for it in range(nIt):
        Vprev = Vs[-1]

        V = np.copy(Vprev)  # Initialize the new value function
        pi = np.zeros(mdp.nS, dtype=int)  # Initialize the new policy

        for state in range(mdp.nS):
            action_values = np.zeros(mdp.nA)
            for action in range(mdp.nA):
                for probability, nextstate, reward in mdp.P[state][action]:
                    action_values[action] += probability * (reward + gamma * Vprev[nextstate])

            V[state] = np.max(action_values)
            pi[state] = np.argmax(action_values)

        # max_diff = np.abs(V - Vprev).max()
        # nChgActions = "N/A" if oldpi is None else (pi != oldpi).sum()

        Vs.append(V)
        pis.append(pi)

    return Vs, pis


GAMMA = 0.95
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20)

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