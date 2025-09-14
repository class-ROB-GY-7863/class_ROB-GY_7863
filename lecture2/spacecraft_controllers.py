
import numpy as np 
from spacecraft_dynamics import *

def empty_controller(x):
	return np.zeros(6,)


def grav_comp_pd_attitude_controller(x):
    Kp = 10*np.diag([10.0, 10.0, 10.0])
    Kd = 1*np.diag([5.0, 5.0, 5.0])
    euler_d = np.array([0,0,0])
    R_d = R_from_euler_zyx(*euler_d)
    omega_d = np.zeros(3,)
    R = R_from_euler_zyx(*x[3:6])
    error = 1/2 * skew_matrix_to_vector(R_d.T @ R - R.T @ R_d)
    error_rate = x[9:12] - R.T @ R_d @ omega_d
    u = np.zeros(6,)
    u[3:6] = - Kp @ error - Kd @ x[9:12]
    # print("u",u)
    return u 


def so3_pd_attitude_controller(x, param, euler_d=(0.0, 0.0, 0.0), omega_d=np.zeros(3)):
    """
    x: Newton-Euler state [r, (phi,theta,psi), v, omega_body]
    param: dict with "inertia" (3x3)
    """
    I = param["inertia"]
    # Desired attitude & rates
    R_d = R_from_euler_zyx(*euler_d)
    # Current attitude & rates
    R = R_from_euler_zyx(*x[3:6])                 # body->world
    omega = x[9:12]                                # body frame
    # SO(3) attitude error (Tayebi/Lee et al.)
    e_R = 0.5 * skew_matrix_to_vector(R_d.T @ R - R.T @ R_d)
    # Angular velocity error
    e_omega = omega - (R.T @ R_d @ omega_d)        # body frame

    # Diagonal gains (Nm/rad and Nm/(rad/s))
    # Kp = 15*np.diag([50.0, 50.0, 30.0]) 
    # Kd = 15*np.diag([100.0, 100.0, 60.0]) 
    Kp = 0.1*np.diag([50.0, 50.0, 30.0]) 
    Kd = 0.1*np.diag([100.0, 100.0, 60.0]) 

    # Computed-torque PD (cancels -ω×Iω term in dynamics)
    tau = - Kp @ e_R - Kd @ e_omega + np.cross(omega, I @ omega)

    # (Optional) clip to keep MuJoCo from hard-clipping after ND conversion
    # tau = np.clip(tau, -tau_max, tau_max)

    u = np.zeros(6)
    u[3:6] = tau
    return u