start_fe = 0.1;
end_fe = 0.6;
start_rk = 0;
end_rk = 0.5;
number = 6;
"""
SIR disease model

S' = -beta*S*I
I' = beta*S*I - nu*I
R' = nu*I
"""

import numpy as np;
import RungeKutt4;
import ForwardEuler
from matplotlib import pyplot as plt;

def f(u, t):
    return u + 2*t - 3

def u_analytical(t):
    return 2*np.exp(t) - 2*t + 1

U0 = 3
time_points_fe, time_points_rk =np.linspace(start_fe, end_fe, number), np.linspace(start_rk, end_rk, number)

fe_solver = ForwardEuler.ForwardEuler(f)
fe_solver.set_initial_conditions(U0)
u_fe, t_fe = fe_solver.solve(time_points_fe, False)

rk_solver = RungeKutt4.RungeKutt4(f)
rk_solver.set_initial_conditions(U0)
u_rk, t_rk = rk_solver.solve(time_points_rk, True)

u_analytical_vals = u_analytical(t_rk)

fe_errors = np.abs(u_fe[:, 0] - u_analytical_vals)
rk_errors = np.abs(u_rk[:, 0] - u_analytical_vals)

header = f"{'X':<10}{'M-д Eйлера':<15}{'M-д Рунге-Кутта':<20}{'Точне значення':<20}{'Похибка м-ду Eйлера':<25}{'Похибка м-ду Рунге-Кутта':<25}"
print(header)

for i in range(len(time_points_rk)):
    print(f"{time_points_rk[i]:<10.2f}{u_fe[i, 0]:<15.5f}{u_rk[i, 0]:<20.5f}{u_analytical_vals[i]:<20.5f}{fe_errors[i]:<25.5f}{rk_errors[i]:<25.5f}")

plt.figure(figsize=(10, 6))
plt.plot(t_rk, u_fe[:, 0], 'g^-', label='Forward Euler')
plt.plot(t_rk, u_rk[:, 0], 'bo-', label='Runge-Kutta 4')
plt.plot(t_rk, u_analytical_vals, 'r--', label='Analytical Solution')
plt.title('Comparison of Numerical Methods and Analytical Solution')
plt.xlabel('Time (t)')
plt.ylabel('U(t)')
plt.legend()
plt.grid(True)
plt.show()