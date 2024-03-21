import ODESolver

class ForwardEuler(ODESolver.ODESolver):
    def advance(self):
        u,f,i,time_points = self.u, self.f, self.i, self.time_points;
        dt = time_points[i+1] - time_points[i];
        return u[i, :] + (dt * f(u[i, :], time_points[i]));