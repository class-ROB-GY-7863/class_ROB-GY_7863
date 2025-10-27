
import numpy as np
from scipy.linalg import solve_discrete_are


# controllers
class Controller: 
    def __call__(self, x):
        # return u in [m,1]
        raise NotImplementedError

class EmptyController(Controller):
    def __call__(self, x, rng, system, scale=1.0):
        u = np.zeros((system.m, 1))
        return u

class LQRController(Controller):
    def __call__(self, x, rng, system):
        P = solve_discrete_are(system.A, system.B, system.Q_x, system.Q_u)
        K = np.linalg.inv(system.Q_u + system.B.T @ P @ system.B) @ (system.B.T @ P @ system.A)
        u = -K @ x
        return u

class MPPI(Controller):
    def __call__(self, x, rng, system):
        raise NotImplementedError