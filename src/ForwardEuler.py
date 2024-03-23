import ODESolver
import numpy as np

class ForwardEuler(ODESolver.ODESolver):
    def advance(self):
        u,f,i,time_points = self.u, self.f, self.i, self.time_points;
        eps = 0.001
        dt = time_points[i+1] - time_points[i];
        t = time_points[i]

        if not self.is_adaptive:
            
            return u[i, :] + (dt * f(u[i, :], time_points[i]));
    
        while True:
            u1 = u[i, :] + (dt * f(u[i, :], t))

            dt_half = dt / 2.0
            u_temp = u[i, :] + (dt_half * f(u[i, :], t))
            u2 = u_temp + (dt_half * f(u_temp, t + dt_half))

            if np.linalg.norm(u1 - u2) < eps:
                return u2
            else:
                dt /= 2