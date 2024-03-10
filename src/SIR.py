import numpy as np

class SIR:
    def __init__(self, nu, beta, S0, I0, R0):
        """
        nu, beta - params in ODE system
        S0, I0, R0 - initial values
        """

        if (isinstance(nu, (float, int))):
            self.nu = lambda t: nu; 
        elif callable(nu):
            self.nu = nu;

        if (isinstance(beta, (float, int))):
            self.beta = lambda t: beta; 
        elif callable(beta):
            self.beta = beta;
        
        self.initial_conditions = [S0,I0,R0];

    def __call__(self, u, t):
        S, I, _ = u 
        return np.asarray([
            -self.beta(t) * S * I, # Susceptibles
            self.beta(t) * S * I - self.nu(t) * I, # Infected
            self.nu(t) * I, # Recovered
        ]);
