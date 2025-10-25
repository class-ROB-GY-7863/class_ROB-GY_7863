

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


# estimators
class Estimator:
    def estimate(
        self,
        xhat_k: np.ndarray,
        P_k: np.ndarray,
        u_kp1: np.ndarray,
        y_kp1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class Kalman(Estimator):
    def __init__(self, system):
        self.sys = system

    def estimate(
        self,
        xhat_k: np.ndarray,
        P_k: np.ndarray,
        u_kp1: np.ndarray,
        y_kp1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        A, B, C, W, V = self.sys.A, self.sys.B, self.sys.C, self.sys.W, self.sys.V

        # Predict
        xhat_prior = A @ xhat_k + B @ u_kp1
        P_prior = A @ P_k @ A.T + W

        # Innovation
        S = C @ P_prior @ C.T + V
        K = P_prior @ C.T @ np.linalg.inv(S)
        y_pred = C @ xhat_prior
        innov = y_kp1 - y_pred

        # Update
        xhat_post = xhat_prior + K @ innov
        I = np.eye(self.sys.n())
        P_post = (I - K @ C) @ P_prior
        return xhat_post, P_post


# controllers
class Controller: 
    def __call__(self, x):
        # return u in [m,1]
        raise NotImplementedError

class RandomController(Controller):
    def __init__(self, system):
        self.m = system.m()

    def __call__(self, x, rng, scale=1.0):
        u = rng.normal(loc=0.0, scale=scale, size=(self.m, 1))
        return u

class FeedbackController(Controller):
    def __init__(self, system):
        self.m = system.m()
        self.K = np.diag(np.ones(max(system.m(), system.n())))
        self.K = self.K[0:system.m(),0:system.n()]

    def __call__(self, x, rng, scale=1.0):
        u = -self.K @ x
        return u

class EmptyController(Controller):
    def __init__(self, system):
        self.m = system.m()

    def __call__(self, x, rng, scale=1.0):
        u = np.zeros((self.m, 1))
        return u


# systems
class System:
    def dynamics(self, x_k: np.ndarray, u_kp1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return x_{k+1} given x_k and u_{k+1}. x_k: [n,1], u_kp1: [m,1]"""
        raise NotImplementedError

    def measurement(self, x_k: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return y_k given x_k. x_k: [n,1] -> y_k: [p,1]"""
        raise NotImplementedError

    def likelihood(self, y_k: np.ndarray, x_k: np.ndarray) -> float:
        """Return p(y|x) under the measurement model."""
        raise NotImplementedError

    def n(self) -> int:
        raise NotImplementedError

    def m(self) -> int:
        raise NotImplementedError

    def p(self) -> int:
        raise NotImplementedError

@dataclass
class LTI(System):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    W: np.ndarray  # process noise covariance (Q)
    V: np.ndarray  # measurement noise covariance (R)
    P0: np.ndarray # initial covariance (P0)

    @staticmethod
    def default(n: int = 2, m: int = 1, p: int = 1) -> "LTI":
        # A simple lightly-damped oscillator as a default
        A = np.array([[1.0, 0.1],
                      [-0.2, 0.98]])
        B = np.array([[0.0], [0.1]])
        C = np.array([[1.0, 0.0]])
        W = np.diag([1e-3, 1e-3])
        V = np.diag([1e-2])
        P0 = np.diag([1.0, 1.0])
        assert A.shape == (n, n)
        assert B.shape == (n, m)
        assert C.shape == (p, n)
        assert W.shape == (n, n)
        assert V.shape == (p, p)
        return LTI(A=A, B=B, C=C, W=W, V=V, P0=P0)

    @staticmethod
    def random(
        n: int,
        m: int,
        p: int,
        timestep: float,
        p0_scale : float,
        w_scale: float,
        v_scale: float,
        rng: np.random.Generator,
        pole_radius: float = 0.95,
        max_tries: int = 200,
        pd_eps: float = 1e-6,
        ) -> "LTI":
        """
        Sample a stable, controllable, observable discrete-time LTI.
        - A eigenvalues strictly inside the unit circle (scaled by pole_radius).
        - (A,B) controllable and (A,C) observable by resampling until ranks are full.
        - W, V SPD from random factors.
        """
        def controllability_rank(A, B):
            n_ = A.shape[0]
            Ctrb = B
            Ak = np.eye(n_)
            for _ in range(1, n_):
                Ak = Ak @ A
                Ctrb = np.concatenate([Ctrb, Ak @ B], axis=1)
            return np.linalg.matrix_rank(Ctrb)
        def observability_rank(A, C):
            n_ = A.shape[0]
            Obsv = C
            AkT = np.eye(n_)
            for _ in range(1, n_):
                AkT = AkT @ A.T
                Obsv = np.concatenate([Obsv, (AkT @ C.T).T], axis=0)
            return np.linalg.matrix_rank(Obsv)
        for _ in range(max_tries):
            # Random A then scale to desired spectral radius (ensures stability)
            Ac = rng.normal(size=(n, n))
            A = Ac * timestep + np.eye(n)
            spec_rad = max(1e-9, max(abs(np.linalg.eigvals(A))))
            A = (pole_radius / (1.05 * spec_rad)) * A # margin inside disk
        B = rng.normal(size=(n, m))
        C = rng.normal(size=(p, n))
        if controllability_rank(A, B) == n and observability_rank(A, C) == n:
            Gw = rng.normal(size=(n, n))
            Gv = rng.normal(size=(p, p))
            Gp = rng.normal(size=(n, n))
            W = Gw @ Gw.T * w_scale + pd_eps * np.eye(n)
            V = Gv @ Gv.T * v_scale + pd_eps * np.eye(p)
            P0 = p0_scale * np.eye(n)
            return LTI(A=A, B=B, C=C, W=W, V=V, P0=P0)
        raise RuntimeError("Failed to sample controllable & observable LTI in max_tries.")

    def dynamics(self, x_k: np.ndarray, u_kp1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        w_k = rng.multivariate_normal(mean=np.zeros(self.n()), cov=self.W).reshape(-1, 1)
        x_kp1 = self.A @ x_k + self.B @ u_kp1 + w_k
        return x_kp1

    def measurement(self, x_k: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        v_k = rng.multivariate_normal(mean=np.zeros(self.p()), cov=self.V).reshape(-1, 1)
        y_k = self.C @ x_k + v_k
        return y_k

    def likelihood(self, y_k: np.ndarray, x_k: np.ndarray) -> float:
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

    def n(self) -> int:
        return self.A.shape[0]

    def m(self) -> int:
        return self.B.shape[1]

    def p(self) -> int:
        return self.C.shape[0]


def simulate(p: Dict[str, Any], rng):

    num_timesteps = int(p.get("num_timesteps", 100))

    system_name = p.get("system_name", "LTI")
    estimator_name = p.get("estimator_name", "Kalman")
    controller_name = p.get("controller_name", "RandomController")

    # system
    if system_name == "LTI":
        if "system_parameters" in p.keys():
            system = LTI.random(*p["system_parameters"], rng)
        else:
            system = LTI.default()
    else:
        raise ValueError(f"Unsupported system: {system_name}")

    # estimator
    if estimator_name == "Kalman":
        estimator = Kalman(system)
    else:
        raise ValueError(f"Unsupported estimator: {estimator_name}")

    # controller
    if controller_name == "RandomController":
        controller = RandomController(system)
    elif controller_name == "FeedbackController":
        controller = FeedbackController(system)
    elif controller_name == "EmptyController":
        controller = EmptyController(system)
    else:
        raise ValueError(f"Unsupported controller: {controller_name}")

    # Initial conditions
    x_0 = np.zeros((system.n(), 1))  # true initial state
    P_0 = system.P0
    xhat_0 = rng.multivariate_normal(mean=x_0[:,0], cov=P_0).reshape(-1, 1)
    print("P_0",P_0)
    print("x_0",x_0)
    print("xhat_0",xhat_0)

    # data 
    xs, us, ys, xhats, Ps = [x_0], [], [], [xhat_0], [P_0]

    # loop 
    for _ in range(num_timesteps):
        u_kp1 = controller(xs[-1], rng)
        x_kp1 = system.dynamics(xs[-1], u_kp1, rng)
        y_kp1 = system.measurement(x_kp1, rng)
        xhat_kp1, P_kp1 = estimator.estimate(xhats[-1], Ps[-1], u_kp1, y_kp1)

        xs.append(x_kp1)
        us.append(u_kp1)
        ys.append(y_kp1)
        xhats.append(xhat_kp1)
        Ps.append(P_kp1)

    # save data 
    data_fn = p.get("data_fn", "./estimator_data.npz")
    xs_arr = np.stack(xs, axis=0)  # [T, n, 1]
    us_arr = np.stack(us, axis=0)  # [T, m, 1]
    ys_arr = np.stack(ys, axis=0)  # [T, p, 1]
    xhats_arr = np.stack(xhats, axis=0)  # [T, n, 1]
    Ps_arr = np.stack(Ps, axis=0)  # [T, n, n]
    np.savez(
        data_fn,
        xs=xs_arr,
        us=us_arr,
        ys=ys_arr,
        xhats=xhats_arr,
        Ps=Ps_arr,
        A=system.A,
        B=system.B,
        C=system.C,
        W=system.W,
        V=system.V,
        meta=np.array([system.n(), system.m(), system.p(), seed, num_timesteps], dtype=float),
    )
    print(f"Saved simulation to {data_fn}")


def visualize(p: Dict[str, Any]):
    data_fn = p.get("data_fn", "./estimator_data.npz")
    data = np.load(data_fn, allow_pickle=True)

    xs = data["xs"]  # [T, n, 1]
    xhats = data["xhats"]  # [T, n, 1]
    Ps = data["Ps"]  # [T, n, n]

    T = xs.shape[0]
    n = xs.shape[1]
    t = np.arange(T)

    fig, axs = plt.subplots(ncols=4, nrows=n//4 + 1, squeeze=False)
    for i in range(n):
        row = i // 4 
        col = i % 4
        ax = axs[row, col]
        ax.plot(t, xs[:, i, 0], label=f"x[{i}] (true)")
        ax.plot(t, xhats[:, i, 0], linestyle="--", label=f"xhat[{i}] (estimate)")
        ax.set_xlabel("timestep")
        ax.set_ylabel(f"state {i}")
        ax.legend()

    # Error norms
    err = (xhats - xs).reshape(T, n)
    l2 = np.linalg.norm(err, axis=1)
    plt.figure()
    plt.plot(t, l2)
    plt.xlabel("timestep")
    plt.ylabel(r"$\|\hat{x} - x\|_2$")
    plt.title("Estimation error norm over time")
    plt.tight_layout()
    plt.show()

    if n == 1: 
        # special vis 
        pass 



# n: int,
# m: int,
# p: int,
# timestep: float,
# p0_scale : float,
# w_scale: float,
# v_scale: float,
system_parameters_1 = [1, 1, 1, 0.01, 100, 1.0, 1.0]
# system_parameters_1 = [5, 2, 2, 0.01, 100, 1.0, 1.0]


if __name__ == "__main__":

    seed = 0
    rng = np.random.default_rng(seed)

    p = {
        "num_timesteps": 100,
        "timestep" : 0.01, 
        "system_name": "LTI",
        "system_parameters" : system_parameters_1,
        "estimator_name": "Kalman",
        # "controller_name": "RandomController",
        # "controller_name": "FeedbackController",
        "controller_name": "EmptyController",
        "data_fn": "./estimator_data.npz",
    }

    simulate(p, rng)
    visualize(p)
