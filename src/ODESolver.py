import numpy as np

class ODESolver: 
    """
    Any class inheriting from this class should implement 'advance' method
    """
    def __init__(self,f) -> None:
        self.f = f;

    def advance(self):
        """Advance solution one time step"""
        raise NotImplementedError
    
    def set_initial_conditions(self, U0):
        if (isinstance(U0, (int,float))):
            # Scalar ODE
            self.number_of_equations = 1;
            U0 = float(U0);
        else: 
            # System of equations
            U0 = np.asarray(U0);
            self.number_of_equations = U0.size;
        
        self.U0 = U0;

    def solve(self, time_points):
        self.time_points = np.asarray(time_points);
        n = self.time_points.size;

        self.u = np.zeros((n, self.number_of_equations));
        self.u[0, :] = self.U0;

        # Integrate  
        for i in range(n - 1):
            self.i = i;
            self.u[i+1] = self.advance();
        
        return self.u[:i+2], self.time_points[:i+2];

        
