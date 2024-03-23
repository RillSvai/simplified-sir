import ODESolver
import numpy as np
class RungeKutt4(ODESolver.ODESolver):
    def advance(self):
        u, f, i, time_points = self.u, self.f, self.i, self.time_points
        dt = time_points[i+1] - time_points[i]
        t = time_points[i]
        if not self.is_adaptive:
            
            dt_half = dt / 2.0

            k1 = f(u[i, :], t)
            k2 = f(u[i, :] + dt_half * k1, t + dt_half)
            k3 = f(u[i, :] + dt_half * k2, t + dt_half)
            k4 = f(u[i, :] + dt * k3, t + dt)

            return u[i, :] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        eps = 0.001
        dt = time_points[i+1] - time_points[i]
        while True:
            dt_half = dt / 2.0

            k1 = f(u[i, :], t)
            k2 = f(u[i, :] + dt_half * k1, t + dt_half)
            k3 = f(u[i, :] + dt_half * k2, t + dt_half)
            k4 = f(u[i, :] + dt * k3, t + dt)
            u1 = u[i, :] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            k1 = f(u[i, :], t)
            k2 = f(u[i, :] + dt_half / 2.0 * k1, t + dt_half / 2.0)
            k3 = f(u[i, :] + dt_half / 2.0 * k2, t + dt_half / 2.0)
            k4 = f(u[i, :] + dt_half * k3, t + dt_half)
            first_half = u[i, :] + (dt_half / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            k1 = f(first_half, t + dt_half)
            k2 = f(first_half + dt_half / 2.0 * k1, t + dt_half + dt_half / 2.0)
            k3 = f(first_half + dt_half / 2.0 * k2, t + dt_half + dt_half / 2.0)
            k4 = f(first_half + dt_half * k3, t + dt)
            second_half = first_half + (dt_half / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            if np.linalg.norm(second_half - u1) < eps:
                return second_half
            else:
                dt /= 2  