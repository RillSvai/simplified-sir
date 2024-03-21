import ODESolver

class RungeKutt4(ODESolver.ODESolver):
    def advance(self):
        u, f, i, time_points = self.u, self.f, self.i, self.time_points
        dt = time_points[i+1] - time_points[i]
        t = time_points[i]
        dt_half = dt / 2.0

        k1 = f(u[i, :], t)
        k2 = f(u[i, :] + dt_half * k1, t + dt_half)
        k3 = f(u[i, :] + dt_half * k2, t + dt_half)
        k4 = f(u[i, :] + dt * k3, t + dt)

        return u[i, :] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
