
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
    Q_u = np.diag([1e-1]),
    Sigma_x = np.diag([1e-2, 1e-2]),
    Sigma_y = np.diag([1e-2]),
    x_0 = np.array([[10.0], [10.0]]),
    P_0 = np.diag([10000.0, 10000.0]))


# @dataclass
# class LTI(System):
#     A: np.ndarray
#     B: np.ndarray
#     C: np.ndarray
#     Sigma_x: np.ndarray  
#     Sigma_y: np.ndarray 
#     P0: np.ndarray # initial covariance (P0)

#     @staticmethod
#     def default(n: int = 2, m: int = 1, p: int = 1) -> "LTI":
#         # A simple lightly-damped oscillator as a default
#         A = np.array([[1.0, 0.1],
#                       [-0.2, 0.98]])
#         B = np.array([[0.0], [0.1]])
#         C = np.array([[1.0, 0.0]])
#         Sigma_x = np.diag([1e-3, 1e-3])
#         Sigma_y = np.diag([1e-2])
#         P0 = np.diag([1.0, 1.0])
#         assert A.shape == (n, n)
#         assert B.shape == (n, m)
#         assert C.shape == (p, n)
#         assert W.shape == (n, n)
#         assert V.shape == (p, p)
#         return LTI(A=A, B=B, C=C, Sigma_x=Sigma_x, Sigma_y=Sigma_y, P0=P0)

#     @staticmethod
#     def random(
#         n: int,
#         m: int,
#         p: int,
#         timestep: float,
#         p0_scale : float,
#         sigma_x_sqrd_scale: float,
#         sigma_y_sqrd_scale: float,
#         rng: np.random.Generator,
#         pole_radius: float = 0.95,
#         max_tries: int = 200,
#         pd_eps: float = 1e-6,
#         ) -> "LTI":
#         """
#         Sample a controllable and observable discrete-time LTI.
#         - (A,B) controllable and (A,C) observable by resampling until ranks are full.
#         - W, V SPD from random factors.
#         """
#         def controllability_rank(A, B):
#             n_ = A.shape[0]
#             Ctrb = B
#             Ak = np.eye(n_)
#             for _ in range(1, n_):
#                 Ak = Ak @ A
#                 Ctrb = np.concatenate([Ctrb, Ak @ B], axis=1)
#             return np.linalg.matrix_rank(Ctrb)
#         def observability_rank(A, C):
#             n_ = A.shape[0]
#             Obsv = C
#             AkT = np.eye(n_)
#             for _ in range(1, n_):
#                 AkT = AkT @ A.T
#                 Obsv = np.concatenate([Obsv, (AkT @ C.T).T], axis=0)
#             return np.linalg.matrix_rank(Obsv)
#         for _ in range(max_tries):
#             # Random A then scale to desired spectral radius (ensures stability)
#             Ac = rng.normal(size=(n, n))
#             A = Ac * timestep + np.eye(n)
#         B = rng.normal(size=(n, m))
#         C = rng.normal(size=(p, n))
#         if controllability_rank(A, B) == n and observability_rank(A, C) == n:
#             Sigma_x = sigma_x_sqrd_scale * np.eye(n)
#             Sigma_y = sigma_y_sqrd_scale * np.eye(p)
#             P0 = p0_scale * np.eye(n)
#             return LTI(A=A, B=B, C=C, Sigma_x=Sigma_x, Sigma_y=Sigma_y, P0=P0)
#         raise RuntimeError("Failed to sample controllable & observable LTI in max_tries.")

#     def dynamics(self, x_k: np.ndarray, u_kp1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
#         w_k = rng.multivariate_normal(mean=np.zeros(self.n()), cov=self.Sigma_x).reshape(-1, 1)
#         x_kp1 = self.A @ x_k + self.B @ u_kp1 + w_k
#         return x_kp1

#     def measurement(self, x_k: np.ndarray, rng: np.random.Generator) -> np.ndarray:
#         v_k = rng.multivariate_normal(mean=np.zeros(self.p()), cov=self.Sigma_y).reshape(-1, 1)
#         y_k = self.C @ x_k + v_k
#         return y_k

#     def likelihood(self, y_k: np.ndarray, x_k: np.ndarray) -> float:
#         # Gaussian measurement: y | x ~ N(Cx, V)
#         mu = (self.C @ x_k).reshape(-1)
#         y = y_k.reshape(-1)
#         p = self.p()
#         detV = float(np.linalg.det(self.V))
#         invV = np.linalg.inv(self.V)
#         diff = (y - mu)
#         expo = -0.5 * diff.T @ invV @ diff
#         norm = 1.0 / np.sqrt(((2.0 * np.pi) ** p) * detV)
#         return float(norm * np.exp(expo))

#     def n(self) -> int:
#         return self.A.shape[0]

#     def m(self) -> int:
#         return self.B.shape[1]

#     def p(self) -> int:
#         return self.C.shape[0]
