
import numpy as np 
from dataclasses import dataclass

# systems
class System:
    def dynamics(self, x_k, u_kp1, rng): raise NotImplementedError
    def measurement(self, x_k, rng): raise NotImplementedError
    def likelihood(self, y_k, x_k): raise NotImplementedError
    def n(self): raise NotImplementedError
    def m(self): raise NotImplementedError
    def p(self): raise NotImplementedError

@dataclass
class LTI(System):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q_x: np.ndarray  
    Q_u: np.ndarray  
    Sigma_x: np.ndarray  
    Sigma_y: np.ndarray 
    x_0: np.ndarray
    P_0: np.ndarray

    def dynamics(self, x_k, u_kp1, rng):
        w_k = rng.multivariate_normal(mean=np.zeros(self.n()), cov=self.Sigma_x).reshape(-1, 1)
        x_kp1 = self.A @ x_k + self.B @ u_kp1 + w_k
        return x_kp1

    def measurement(self, x_k, rng):
        v_k = rng.multivariate_normal(mean=np.zeros(self.p()), cov=self.Sigma_y).reshape(-1, 1)
        y_k = self.C @ x_k + v_k
        return y_k

    def likelihood(self, y_k, x_k):
        # Gaussian measurement: y | x ~ N(Cx, V)
        mu = (self.C @ x_k).reshape(-1)
        y = y_k.reshape(-1)
        p = self.p()
        detV = float(np.linalg.det(self.V))
        invV = np.linalg.inv(self.V)
        diff = (y - mu)
        expo = -0.5 * diff.T @ invV @ diff
        norm = 1.0 / np.sqrt(((2.0 * np.pi) ** p) * detV)
        return float(norm * np.exp(expo))

    def n(self): return self.A.shape[0]
    def m(self): return self.B.shape[1]
    def p(self): return self.C.shape[0]


LTI_1 = LTI(
    A = np.array([[1.0, 0.1], [-0.2, 0.98]]),
    B = np.array([[0.0], [0.1]]),
    C = np.array([[1.0, 0.0]]),
    Q_x = np.diag([1e-1, 1e-1]),
    Q_u = np.diag([1e-2]),
    Sigma_x = np.diag([1e-2, 1e-2]),
    Sigma_y = np.diag([1e-2]),
    x_0 = np.array([[10.0], [10.0]]),
    P_0 = np.diag([10000.0, 10000.0]))


LTI_2 = LTI(
    A = np.array([[1.0, -0.5], [-0.5, 1.0]]),
    B = np.array([[0.0], [0.1]]),
    C = np.array([[1.0, 0.0]]),
    Q_x = np.diag([1e-1, 1e-1]),
    Q_u = np.diag([1e-2]),
    Sigma_x = np.diag([1e-2, 1e-2]),
    Sigma_y = np.diag([1e-2]),
    x_0 = np.array([[10.0], [10.0]]),
    P_0 = np.diag([10000.0, 10000.0]))


class MJXPendulum(System): 
    def __init__(self):
        raise NotImplementedError