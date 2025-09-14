
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
        "dt" : 0.1,          # physical outer step (s)
        "mj_dt" : 0.001,     # physical inner step target (s)  << smaller is safer
    }

    # ---- ND scales ----
    nd = make_nd_scales(param)   # dict(L0,T0,M0,V0,F0,TAU0)

    # ---- MuJoCo load ----
    xml_path = os.path.join(os.path.dirname(__file__), "./spacecraft_nd.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # ---- Time stepping in ND ----
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    # mj_dt_nd = 1e-4
    # mj_dt_nd = 1e-2
    model.opt.timestep = mj_dt_nd
    substeps = 1
    dt = mj_dt_nd * nd["T0"]

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spacecraft")
    I_nd = inertia_phys_to_nd(param["inertia"], nd)
    model.body_mass[bid] = 1.0
    model.body_inertia[bid] = np.array([I_nd[0,0], I_nd[1,1], I_nd[2,2]])

    # ---- Initial conditions (ND in MuJoCo) ----
    x0 = param["x0"]
    mujoco.mj_resetData(model, data)
    qpos_nd, qvel_nd = newton_to_mujoco_x_nd(x0, nd)
    data.qpos[:] = qpos_nd
    data.qvel[:] = qvel_nd
    mujoco.mj_forward(model, data)

    # ---- Logs (MuJoCo saved in PHYSICAL units for plotting) ----
    xs_mujoco   = [mujoco_to_newton_x_from_nd(data.qpos, data.qvel, nd)]
    xs_lagrange = [newton_to_lagrange_x(x0)]
    xs_newton   = [x0]

    # simulate for 128 minutes physical time, using snapped dt_phys
    num_steps = int((128 * 60) / dt)
    print("num_steps",num_steps)

    # controller in PHYSICAL units: u = [f_world, tau_body]
    controller = lambda x: np.zeros(6,)

    viewer = mujoco.viewer.launch_passive(model, data) if param["mujoco_viewer"] else util.DummyContext()

    with viewer:
        for t in range(num_steps):
            print(f"step/total: {t}/{num_steps}")

            # control: physical -> ND once per outer step
            u = controller(None)
            u_nd = u_to_nd(u, nd)

            # integrate MuJoCo 'substeps' times in ND
            for k in range(substeps):
                # ND gravity at current ND position
                f_g_nd = force_gravity_world_nd(data.qpos[:3])
                # world force (ND) + commanded
                data.ctrl[0:3] = u_nd[0:3] + f_g_nd
                # world torque (ND) from body torque
                data.ctrl[3:6] = R_from_quat(data.qpos[3:7]) @ u_nd[3:6]

                mujoco.mj_step(model, data)

            # log MuJoCo back to PHYSICAL units
            x_mj = mujoco_to_newton_x_from_nd(data.qpos, data.qvel, nd)

            # step your PHYSICAL solvers with the snapped dt
            x_lag = xs_lagrange[-1] + dt * dxdt_lagrange(xs_lagrange[-1], u, param)
            x_new = xs_newton[-1] + dt * dxdt_newton(xs_newton[-1], u, param)

            xs_mujoco.append(x_mj)
            xs_lagrange.append(x_lag)
            xs_newton.append(x_new)

            if param["mujoco_viewer"]:
                viewer.sync()

    # numpy arrays (PHYSICAL units)
    xs_mujoco   = np.array(xs_mujoco)
    xs_lagrange = np.array(xs_lagrange)
    xs_newton   = np.array(xs_newton)

    util.save_h5py(fn, {"xs_mujoco": xs_mujoco,
                        "xs_lagrange": xs_lagrange,
                        "xs_newton": xs_newton}, param=param)


def plot(fn):
    np_dict, param = util.load_h5py(fn)
    xs_mujoco = np_dict["xs_mujoco"]
    xs_newton = np_dict["xs_newton"]
    xs_lagrange = np_dict["xs_lagrange"]
    state_labels = ["r_x","r_y","r_z","phi","theta","psi","v_x","v_y","v_z","omega_x","omega_y","omega_z"]
    nrows, ncols = 3, 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,15))
    for ii_row in range(nrows):
        for ii_col in range(ncols):
            ax_ij = ax[ii_row, ii_col]
            ax_ij.plot(xs_mujoco[:,ii_row*ncols+ii_col], linestyle=(0, (2, 6)), label="MuJoCo")
            ax_ij.plot(xs_lagrange[:,ii_row*ncols+ii_col], linestyle=(2, (2, 6)), label="Euler-Lagrange")
            ax_ij.plot(xs_newton[:,ii_row*ncols+ii_col], linestyle=(4, (2, 6)), label="Newton-Euler")
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














# def simulate(fn):

#     param = {
#         "mass" : 1600, # kg 
#         "radius" : 2.4, # m 
#         "height" : 6.2, # m
#         "inertia" : np.diag([7429.33, 7429.33, 4608.0]), # kg m2 (cylinder model)
#         "mass_earth" : 5.97219e24, # kg
#         "r_earth" : np.array([0,0,0]), # coordinate of earth 
#         "radius_earth" : 6371e3, # m
#         "G" : 6.67430e-11, 
#         "x0" : np.array([6.371e6+500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7670, 1000, 0.001, 0, 0]), # initial state
#         "mujoco_viewer" : False,
#         "realtime_rate" : 100, 
#         "dt" : 0.1,
#         "mj_dt" : 0.1, 
#     }

#     # ---- ND scales ----
#     nd = make_nd_scales(param)  # adds L0,T0,M0,V0,F0,TAU0
    
#     # Path to your MJCF file
#     xml_path = os.path.join(os.path.dirname(__file__), "./spacecraft_nd.xml")

#     # Load model and data
#     model = mujoco.MjModel.from_xml_path(xml_path)
#     data = mujoco.MjData(model)
    
#     # integer number of internal steps per outer step
#     substeps = max(1, int(round(param["dt"] / param["mj_dt"])))
#     model.opt.timestep = param["mj_dt"]
#     model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

#     # Set initial conditions
#     x0 = param["x0"]
#     mujoco.mj_resetData(model, data)
#     qpos_nd, qvel_nd = newton_to_mujoco_x_nd(x0, nd)
#     data.qpos[:] = qpos_nd
#     data.qvel[:] = qvel_nd
#     mujoco.mj_forward(model, data)  # recompute positions, forces

#     # record states
#     xs_mujoco = [np.concatenate([data.qpos, data.qvel])]
#     xs_lagrange = [newton_to_lagrange_x(x0)]
#     xs_newton = [x0]

#     # simulate for 128 minutes of real time (LEO orbit)
#     num_steps = int(128 * 60 / param["dt"])

#     # set controller 
#     controller = lambda x: np.zeros(6,) # zero torque controller

#     # get viewer
#     if param["mujoco_viewer"]:
#         viewer = mujoco.viewer.launch_passive(model, data)
#     else:
#         viewer = util.DummyContext()

#     # run simulation
#     with viewer:

#         for t in range(num_steps):
#             print(f"step/total: {t}/{num_steps}")

#             start = time.time()

#             # compute controller 
#             u = controller(None)
#             u_nd = u_to_nd(u, nd)

#             # Advance physics
#             for k in range(substeps):
#                 data.ctrl[0:3] = u_nd[0:3] + force_gravity_world_nd(data.qpos[0:3]) # world force 
#                 data.ctrl[3:6] = R_from_quat(data.qpos[3:7]) @ u_nd[3:6]  # from body torque to world torque
#                 mujoco.mj_step(model, data)
#             x_mujoco = np.concatenate([data.qpos[:], data.qvel[:]])
            
#             x_lagrange = xs_lagrange[-1] + param["dt"] * dxdt_lagrange(xs_lagrange[-1], u, param) 
#             x_newton = xs_newton[-1] + param["dt"] * dxdt_newton(xs_newton[-1], u, param) 

#             # record states
#             xs_mujoco.append(x_mujoco)
#             xs_lagrange.append(x_lagrange)
#             xs_newton.append(x_newton)

#             if param["mujoco_viewer"]:
#                 viewer.sync()
#                 time_until_next_step = (model.opt.timestep - (time.time() - start)) / param["realtime_rate"]
#                 if time_until_next_step > 0:
#                     time.sleep(time_until_next_step)

#         # close viewer
#         if param["mujoco_viewer"]:
#             viewer.close()

#     # convert to same coordinate frame 
#     xs_lagrange = [lagrange_to_newton_x(x) for x in xs_lagrange]
#     xs_mujoco = [mujoco_to_newton_x_from_nd(x[0:7],x[7:],nd) for x in xs_mujoco]

#     print("xs_mujoco",xs_mujoco)

#     # save data
#     xs_mujoco = np.array(xs_mujoco)
#     xs_lagrange = np.array(xs_lagrange)
#     xs_newton = np.array(xs_newton)

#     print("xs_mujoco", xs_mujoco.shape)
#     print("xs_lagrange", xs_lagrange.shape)
#     print("xs_newton", xs_newton.shape)

#     util.save_h5py(fn, {"xs_mujoco" : xs_mujoco, 
#                         "xs_lagrange" : xs_lagrange, 
#                         "xs_newton" : xs_newton}, param=param)