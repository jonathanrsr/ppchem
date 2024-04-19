from reaction import Reaction, Batch
""" import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np """

reaction1 = Reaction(['A', 'B', 'C'], [-1, -1, 1], '0.2*A**2*B')
reaction2 = Reaction(['B', 'C', 'D'], [-1, -1, 1], '0.75*B*C')
reaction3 = Reaction(['A', 'D'], [2, -0.5], '0.5*D**2/A')
reaction4 = Reaction(['C', 'B'], [-2, 2], '2*C')

initial_concs = {'A': 1, 'B': 2, 'C': 0, 'D': 1}

batch = Batch(1, [reaction1, reaction2, reaction3, reaction4], initial_concs)

results = batch.evolution(100)



# Pour vérifier
""" def ODE(t, y):
    A, B, C, D = y

    r1 = 0.2*A**2*B
    r2 = 0.75*B*C
    r3 = 0.5*D**2/A
    r4 = 2*C

    dA = -r1 + 2*r3
    dB = -r1 - r2 + 2*r4
    dC = r1 - r2 - 2*r4
    dD = r2 - 0.5*r3

    return dA, dB, dC, dD

initial_concs = [1, 2, 0, 1]
results = solve_ivp(ODE, [0, 100], initial_concs, t_eval = np.linspace(0, 100, 1000))

plt.plot(results.t, results.y.T, linewidth = 1.0)
plt.legend(['A', 'B', 'C', 'D'])
plt.show() """