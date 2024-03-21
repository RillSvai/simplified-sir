"""
SIR disease model

S' = -beta*S*I
I' = beta*S*I - nu*I
R' = nu*I
"""

import numpy as np;
import SIR;
import RungeKutt4;
from matplotlib import pyplot as plt;


beta = lambda t: 0.0005 if t <= 10 else 0.0001
sir = SIR.SIR(0.1, beta, 1500, 1, 0);

solver = RungeKutt4.RungeKutt4(sir);
solver.set_initial_conditions(sir.initial_conditions)

t = np.linspace(0, 60, 1001);
u, t = solver.solve(t);

plt.plot(t, u[:, 0], label="Susceptible")
plt.plot(t, u[:, 1], label="Infected")
plt.plot(t, u[:, 2], label="Recovered")
plt.legend()
plt.show()