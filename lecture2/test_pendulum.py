
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import time 


plt.rcParams['font.size'] = 24 
plt.rcParams['lines.linewidth'] = 4


def el_pendulum_dynamics(x, u):
    theta, theta_dot = x
    torque = u[0]
    # Pendulum parameters
    m = 1.0      # mass (kg)
    l = 1.0      # length (m)
    g = 9.81     # gravity (m/s^2)
    b = 0.0      # damping coefficient
    # Dynamics equations
    theta_ddot = (-b * theta_dot - m * g * l * np.sin(theta) + torque) / (m * l**2)
    return np.array([theta_dot, theta_ddot])


def simulate(fn):

    # Path to your MJCF file
    xml_path = os.path.join(os.path.dirname(__file__), "simple_pendulum_point_mass.xml")

    # Load model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Set initial condition from the keyframe
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.5236   # 30 degrees
    data.qvel[0] = 0.0      # zero angular velocity
    mujoco.mj_forward(model, data)  # recompute positions, forces

    # run simulation 
    T = 10 # total time, seconds 
    x0 = np.array([data.qpos[0], data.qvel[0]])
    controller = lambda x: np.array([0.0])  # zero torque controller
    mj_xs, el_xs = [x0], [x0]

    for t in range(int(T/model.opt.timestep)):

        u = controller(None)

        # apply control
        data.ctrl[0] = u

        # Advance physics
        mujoco.mj_step(model, data)
        xdot = el_pendulum_dynamics(el_xs[-1], u)

        # record states
        mj_xs.append(np.array([data.qpos[0], data.qvel[0]]))
        el_xs.append(el_xs[-1] + model.opt.timestep * xdot)

    # save data
    mj_xs = np.array(mj_xs)
    el_xs = np.array(el_xs)
    save(fn, mj_xs, el_xs)


def save(fn, mj_xs, el_xs):
    with h5py.File(fn, "w") as h5file:
        h5file.create_dataset("mj_xs", data=mj_xs)
        h5file.create_dataset("el_xs", data=el_xs)


def load(fn):
    with h5py.File(fn, "r") as h5file:
        mj_xs = h5file["mj_xs"][:]
        el_xs = h5file["el_xs"][:]
    return mj_xs, el_xs


def plot(fn):
    mj_xs, el_xs = load(fn)
    fig, ax = plt.subplots()
    ax.plot(mj_xs[:,0], label="MuJoCo")
    ax.plot(el_xs[:,0], label="Euler-Lagrange")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Angle (rad)")
    ax.legend()
    plt.show()


def main():
    fn = "./test_pendulum.h5"
    simulate(fn)
    plot(fn)


if __name__ == '__main__':
    main()