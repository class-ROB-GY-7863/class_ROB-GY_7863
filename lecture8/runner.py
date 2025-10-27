
import numpy as np 
from systems import *
from estimators import *
from controllers import *


def simulate(p, rng):

    num_timesteps = int(p.get("num_timesteps", 100))

    system_name = p["system_name"] 
    estimator_name = p["estimator_name"]
    controller_name = p["controller_name"]

    # system
    if system_name == "LTI_1":
        system = LTI_1
    elif system_name == "LTI_2":
        system = LTI_2
    elif system_name == "MJXPendulum":
        system = MJXPendulum
    else:
        raise ValueError(f"Unsupported system: {system_name}")

    # estimator
    if estimator_name == "EmptyEstimator":
        estimator = EmptyEstimator()
    elif estimator_name == "KalmanFilter":
        estimator = KalmanFilter()
    elif estimator_name == "EnsembleKalmanFilter":
        estimator = EnsembleKalmanFilter()
    else:
        raise ValueError(f"Unsupported estimator: {estimator_name}")

    # controller
    if controller_name == "EmptyController":
        controller = EmptyController()
    elif controller_name == "LQRController":
        controller = LQRController()
    elif controller_name == "MPPI":
        controller = MPPI()
    else:
        raise ValueError(f"Unsupported controller: {controller_name}")

    # Initial conditions
    xhat_0 = rng.multivariate_normal(mean=system.x_0[:,0], cov=system.P_0).reshape(-1, 1)

    # data 
    xs, us, ys, xhats, Ps = [system.x_0], [], [], [xhat_0], [system.P_0]

    # loop 
    for _ in range(num_timesteps):
        u_kp1 = controller(xhats[-1], rng, system)
        x_kp1 = system.dynamics(xs[-1], u_kp1, rng)
        y_kp1 = system.measurement(x_kp1, rng)
        xhat_kp1, P_kp1 = estimator.estimate(xhats[-1], Ps[-1], u_kp1, y_kp1, system)

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
        Ps=Ps_arr)
    print(f"Saved simulation to {data_fn}")


def visualize(p):
    data_fn = p.get("data_fn", "./estimator_data.npz")
    data = np.load(data_fn, allow_pickle=True)

    xs = data["xs"] 
    xhats = data["xhats"] 
    Ps = data["Ps"] 

    T = xs.shape[0]
    n = xs.shape[1]
    t = np.arange(T)

    ncols = min(4,n)
    nrows = n//ncols 
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)
    for i in range(n):
        row = i // 4 
        col = i % 4
        ax = axs[row, col]
        ax.plot(t, xs[:, i, 0], label=f"x[{i}]")
        ax.plot(t, xhats[:, i, 0], linestyle="--", label=f"xhat[{i}]")
        ax.set_xlabel("timestep")
        ax.set_ylabel(f"state {i}")
        ax.legend()

    # Error norms
    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False)
    estimator_error = (xhats - xs).reshape(T, n)
    estimator_error_norm = np.linalg.norm(estimator_error, axis=1)
    tracking_error = xs.reshape(T,n)  # assuming reference is zero
    tracking_error_norm = np.linalg.norm(tracking_error, axis=1)
    ax[0,0].plot(t, estimator_error_norm)
    ax[0,0].set_xlabel("timestep")
    ax[0,0].set_ylabel(r"$\|\hat{x} - x\|_2$")
    ax[0,0].set_title("Estimation error")
    ax[0,1].plot(t, tracking_error_norm)
    ax[0,1].set_xlabel("timestep")
    ax[0,1].set_ylabel(r"$\|x - x^{des}\|_2$")
    ax[0,1].set_title("Tracking error")
    plt.show()


if __name__ == "__main__":

    seed = 0
    rng = np.random.default_rng(seed)

    p = {
        "num_timesteps": 100,
        "system_name": "LTI_2",
        "estimator_name": "KalmanFilter",
        "controller_name": "LQRController",
        "data_fn": "./estimator_data.npz",
    }

    simulate(p, rng)
    visualize(p)
