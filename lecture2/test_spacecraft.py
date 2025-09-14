
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import time 

import sys 
sys.path.append("../")
# from lecture1.spacecraft_sim import dxdt_newton
from spacecraft_dynamics import * 
import util 


plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2


# default state convention is Newton-Euler, x = [r, phi, theta, psi, v, omega]
# default action convention is Newton-Euler, u = [f_ext_world, \tau_\ext_body]


def simulate(fn):

    # ---- physical parameters (unchanged) ----
    param = {
        "mass" : 1600, 
        "radius" : 2.4, 
        "height" : 6.2, 
        "inertia" : np.diag([7429.33, 7429.33, 4608.0]),
        "mass_earth" : 5.97219e24,
        "r_earth" : np.array([0,0,0]),
        "radius_earth" : 6371e3,
        "G" : 6.67430e-11, 
        "x0" : np.array([6.371e6+500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7670, 1000, 0.001, 0, 0]),
        "mujoco_viewer" : False,
        "realtime_rate" : 100, 
        "dt" : 0.1, 
        "mj_dt" : 0.1, 
    }

    # controller
    controller = lambda x: np.zeros(6,) # zero torque controller

    # first run newton and lagrange
    xs_lagrange = [newton_to_lagrange_x(param["x0"])]
    xs_newton   = [param["x0"]]
    T = 128 * 60 # typical orbit is 128 minutes
    num_steps = int(T / param["dt"])
    # num_steps = 1 
    for step in range(num_steps):
        if step % 1000 == 0: print(f"step/num_steps: {step}/{num_steps}")
        u = controller(None)
        xs_lagrange.append(xs_lagrange[-1] + param["dt"] * dxdt_lagrange(xs_lagrange[-1], u, param))
        xs_newton.append(xs_newton[-1] + param["dt"] * dxdt_newton(xs_newton[-1], u, param))
    xs_lagrange = np.array(xs_lagrange)
    xs_newton   = np.array(xs_newton)
    ts_newton = np.arange(0, num_steps+1) * param["dt"]

    # now run nondimensional (nd)mujoco 
    nd = make_nd_scales(param)   # dict(L0,T0,M0,V0,F0,TAU0)
    xml_path = os.path.join(os.path.dirname(__file__), "./spacecraft_nd.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    
    # timestep settings
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mj_dt_nd = param["mj_dt"] / nd["T0"]  # nondim dt
    model.opt.timestep = mj_dt_nd

    # inertia settings
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spacecraft")
    I_nd = inertia_phys_to_nd(param["inertia"], nd)
    model.body_mass[bid] = 1.0
    model.body_inertia[bid] = np.array([I_nd[0,0], I_nd[1,1], I_nd[2,2]])

    # initial conditions 
    mujoco.mj_resetData(model, data)
    qpos_nd, qvel_nd = mujoco_nd_from_newton_x(param["x0"], nd)
    data.qpos[:] = qpos_nd
    data.qvel[:] = qvel_nd
    mujoco.mj_forward(model, data)

    # simulate 
    xs_mujoco = [np.concatenate([data.qpos, data.qvel])]
    num_nd_steps = int(T / nd["T0"] / mj_dt_nd) 
    print("num_nd_steps",num_nd_steps)
    viewer = mujoco.viewer.launch_passive(model, data) if param["mujoco_viewer"] else util.DummyContext()
    with viewer:
        for step in range(num_nd_steps):
            if step % 1000 == 0: print(f"step/num_nd_steps: {step}/{num_nd_steps}")
            u = controller(None)
            u_nd = u_phys_to_nd(u, nd)
            f_g_nd = force_gravity_world_nd(data.qpos[:3])
            data.ctrl[0:3] = u_nd[0:3] + f_g_nd
            data.ctrl[3:6] = u_nd[3:6]   
            mujoco.mj_step(model, data)
            xs_mujoco.append(np.concatenate([data.qpos, data.qvel]))
            if param["mujoco_viewer"]:
                viewer.sync()

            # if step > 5: break 
            # print("u_nd",u_nd)
            # print("f_g_nd",f_g_nd)
            # print("data.ctrl",data.ctrl)
            # print("data.qpos",data.qpos)
            # print("data.qvel",data.qvel)


    # numpy arrays (PHYSICAL units)
    xs_mujoco = np.array([newton_x_from_mujoco_nd(x[0:7], x[7:], nd) for x in xs_mujoco])
    ts_mujoco = np.arange(0, num_nd_steps+1) * mj_dt_nd * nd["T0"]

    util.save_h5py(fn, {"xs_mujoco": xs_mujoco,
                        "xs_lagrange": xs_lagrange,
                        "xs_newton": xs_newton,
                        "ts_newton": ts_newton, 
                        "ts_mujoco": ts_mujoco}, param=param)


def plot(fn):
    np_dict, param = util.load_h5py(fn)
    xs_mujoco = np_dict["xs_mujoco"]
    xs_newton = np_dict["xs_newton"]
    xs_lagrange = np_dict["xs_lagrange"]
    ts_mujoco = np_dict["ts_mujoco"]
    ts_newton = np_dict["ts_newton"]
    state_labels = ["r_x","r_y","r_z","phi","theta","psi","v_x","v_y","v_z","omega_x","omega_y","omega_z"]
    nrows, ncols = 3, 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,15))
    for ii_row in range(nrows):
        for ii_col in range(ncols):
            ax_ij = ax[ii_row, ii_col]
            ax_ij.plot(ts_mujoco, xs_mujoco[:,ii_row*ncols+ii_col], linestyle=(0, (2, 6)), label="MuJoCo")
            ax_ij.plot(ts_newton, xs_lagrange[:,ii_row*ncols+ii_col], linestyle=(2, (2, 6)), label="Euler-Lagrange")
            ax_ij.plot(ts_newton, xs_newton[:,ii_row*ncols+ii_col], linestyle=(4, (2, 6)), label="Newton-Euler")
            ax_ij.set_xlabel("Timestep")
            ax_ij.set_ylabel(f"{state_labels[ii_row*ncols+ii_col]}")
            if ii_col == 0 and ii_row == 0:
                ax_ij.legend()
    plt.show()


def main():
    fn = "./test_spacecraft.h5"
    simulate(fn)
    plot(fn)


if __name__ == '__main__':
    main()
