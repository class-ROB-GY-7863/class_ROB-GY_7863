
import numpy as np 

# dynamics 
def dxdt_newton(x, u, param): 
    # x: [r_x,r_y,r_z, phi,theta,psi, v_x,v_y,v_z, omega_x,omega_y,omega_z]
    # u: [f_world_x, f_world_y, f_world_z, tau_body_x, tau_body_y, tau_body_z]
    _dxdt = np.zeros((12,))
    # state
    r = x[0:3] 
    Theta = x[3:6]
    v = x[6:9]
    omega = x[9:12]
    # control 
    f_ext = u[0:3]
    tau_ext = u[3:6]
    # gravity
    r_rel = param["r_earth"] - r
    f_g = param["G"] * param["mass_earth"] * r_rel / np.power(np.linalg.norm(r_rel), 3)
    # dynamics 
    _dxdt[0:3] = v 
    _dxdt[3:6] = Binv_zyx(Theta[0], Theta[1]) @ omega 
    _dxdt[6:9] = f_g + f_ext / param["mass"]
    _dxdt[9:12] = np.linalg.pinv(param["inertia"]) @ (- np.cross(omega, param["inertia"] @ omega) + tau_ext)
    return _dxdt


def dxdt_lagrange(x, u, param):
    # x: [r_x,r_y,r_z, phi,theta,psi, v_x,v_y,v_z, dotphi,dottheta,dotpsi]
    # u: [f_world_x, f_world_y, f_world_z, tau_body_x, tau_body_y, tau_body_z]
    m = param["mass"]
    I = param["inertia"] 
    mu_earth = param["G"] * param["mass_earth"]

    r = x[0:3]
    phi, theta, psi = x[3:6]
    v = x[6:9]
    dth = x[9:12]                     # [dotφ, dotθ, dotψ]
    f_world = u[0:3]
    tau_body = u[3:6]

    # Kinematics
    B = B_zyx(phi, theta)
    omega_body = B @ dth

    # Translation
    r_rel = r - param["r_earth"]
    Rnorm = np.linalg.norm(r_rel)
    g_force = - mu_earth * m * r_rel / (Rnorm**3)   # world frame
    a = (g_force + f_world) / m

    # Attitude dynamics in Euler-rate coordinates
    # J(Θ) = B^T I B
    J = B.T @ I @ B

    # dB/dt = sum_k (∂B/∂Θ_k) dotΘ_k   (only φ,θ enter B)
    # ∂B/∂φ and ∂B/∂θ:
    s, c = np.sin, np.cos
    dB_dphi = np.array([
        [0, 0, 0],
        [0, -s(phi),  c(phi)*c(theta)],
        [0, -c(phi), -s(phi)*c(theta)]
    ])
    dB_dtheta = np.array([
        [0, 0, -c(theta)],
        [0, 0, -s(phi)*s(theta)],
        [0, 0, -c(phi)*s(theta)]
    ])
    Bd = dB_dphi * dth[0] + dB_dtheta * dth[1]

    # CΘ * dotΘ  =  B^T I Bd dotΘ + B^T (ω × (I ω))
    cross = np.array([[0, -omega_body[2], omega_body[1]],
                      [omega_body[2], 0, -omega_body[0]],
                      [-omega_body[1], omega_body[0], 0]])
    Cdot = (B.T @ I @ (Bd @ dth)) + (B.T @ (cross @ (I @ omega_body)))

    # EoM: J * ddotΘ + CΘ*dotΘ = B^T * τ_body   (no conservative attitude torque)
    rhs = (B.T @ tau_body) - Cdot
    ddth = np.linalg.solve(J, rhs)

    # State derivative
    dxdt = np.zeros(12)
    dxdt[0:3] = v
    dxdt[3:6] = dth
    dxdt[6:9] = a
    dxdt[9:12] = ddth
    return dxdt

def dxdt_mujoco(x, u, param, model, data):
	# for free body joint 
    # qpos: (x,y,z)_world, qw, qx, qy, qz 
    # qvel: (omega_x,omega_y,omega_z)_world, (vx, vy, vz)_world
    # u_mujoco: (fx,fy,fz)_world, (tau_x,tau_y,tau_z)_world 
    pass 


def force_gravity_world(r_world, param):
    """Central gravity force (N) on the spacecraft at world position r_world."""
    mu = param["G"] * param["mass_earth"]                   # m^3/s^2
    r_rel = r_world - param["r_earth"]
    R = np.linalg.norm(r_rel)
    if R == 0:
        return np.zeros(3)
    a_g = - mu * r_rel / (R**3)                             # m/s^2
    return param["mass"] * a_g                              # N


# convert angle representations 
def R_from_euler_zyx(phi, theta, psi):
    # R = Rz(psi) * Ry(theta) * Rx(phi) (body->world)
    R_x = np.array([[1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]])
    return R_z @ R_y @ R_x  # Combined rotation matrix


def R_from_quat(q):
    w,x,y,z = q[0],q[1],q[2],q[3]
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])
    return R


def quat_from_R(R):
    # MuJoCo uses (w,x,y,z)
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
        [R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], 0, 0],
        [R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], 0],
        [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], R[0,0]+R[1,1]+R[2,2]]
    ])
    K = K / 3.0
    w, V = np.linalg.eigh(K)
    q = V[:, np.argmax(w)]
    q = np.array([q[3], q[0], q[1], q[2]])  # (w,x,y,z)
    if q[0] < 0: q = -q
    return q/np.linalg.norm(q)


def euler_zyx_from_R(R):
    # Returns (phi, theta, psi) with theta in (-pi/2, pi/2)
    theta = np.arcsin(-R[2,0])
    phi   = np.arctan2(R[2,1], R[2,2])
    psi   = np.arctan2(R[1,0], R[0,0])
    return phi, theta, psi


def B_zyx(phi, theta):
    # ω_body = B(Θ) * [dotφ, dotθ, dotψ]
    s, c = np.sin, np.cos
    return np.array([
        [1, 0, -s(theta)],
        [0, c(phi), s(phi)*c(theta)],
        [0,-s(phi), c(phi)*c(theta)]
    ])


def Binv_zyx(phi, theta):
    # [dotφ, dotθ, dotψ] = B^{-1} * ω_body
    s, c, t = np.sin, np.cos, np.tan
    return np.array([
        [1, s(phi)*t(theta), c(phi)*t(theta)],
        [0, c(phi), -s(phi)],
        [0, s(phi)/c(theta), c(phi)/c(theta)]
    ])


# state conversions
def newton_to_mujoco_x(newton_x):
    """
    Newton state (12) -> MuJoCo freejoint (qpos7, qvel6).
      newton_x = [r(3), euler(3), v_world(3), omega_body(3)]
    qpos = [x y z qw qx qy qz]
    qvel = [ω_world(3) v_world(3)]
    """
    r = newton_x[0:3]
    phi, theta, psi = newton_x[3:6]
    v_world = newton_x[6:9]
    omega_body = newton_x[9:12]

    R = R_from_euler_zyx(phi, theta, psi)   # body->world
    q = quat_from_R(R)                      # (w,x,y,z)
    omega_world = R @ omega_body

    qpos = np.concatenate([r, q])
    qvel = np.concatenate([omega_world, v_world])  # MuJoCo order
    return qpos, qvel


def mujoco_to_newton_x(qpos, qvel):
    """
    MuJoCo freejoint (qpos7, qvel6) -> your Newton state (12) with body rates.
      qpos = [x y z qw qx qy qz]
      qvel = [ω_world(3) v_world(3)]
    Returns x = [r, euler(3), v_world, omega_body]
    """
    r = qpos[0:3]
    q = qpos[3:7]
    R = R_from_quat(q) 
    phi, theta, psi = euler_zyx_from_R(R)

    omega_world = qvel[0:3]
    v_world = qvel[3:6]
    omega_body = R.T @ omega_world

    x = np.concatenate([r, [phi, theta, psi], v_world, omega_body])
    return x


def lagrange_to_newton_x(lagrange_x):
    """
    Lagrange state (Euler-rate last 3) -> Newton state (body ω last 3).
      lagrange_x = [r(3), euler(3), v_world(3), [dotφ,dotθ,dotψ]]
      newton_x   = [r(3), euler(3), v_world(3), omega_body(3)]
    """
    x = lagrange_x.copy()
    r = x[0:3]
    phi, theta, psi = x[3:6]
    v_world = x[6:9]
    dth = x[9:12] 
    B = B_zyx(phi, theta)
    omega_body = B @ dth
    newton_x = np.concatenate([r, [phi, theta, psi], v_world, omega_body])
    return newton_x


def newton_to_lagrange_x(newton_x):
    """
    Lagrange state (Euler-rate last 3) -> Newton state (body ω last 3).
      lagrange_x = [r(3), euler(3), v_world(3), [dotφ,dotθ,dotψ]]
      newton_x   = [r(3), euler(3), v_world(3), omega_body(3)]
    """
    x = newton_x.copy()
    r = x[0:3]
    phi, theta, psi = x[3:6]
    v_world = x[6:9]
    omega = x[9:12] 
    dTheta = Binv_zyx(phi, theta) @ omega
    lagrange_x = np.concatenate([r, [phi, theta, psi], v_world, dTheta])
    return lagrange_x


def newton_to_mujoco_x_nd(x_phys, nd):
    x_nd = x_phys_to_nd(x_phys, nd)
    r = x_nd[0:3]; phi,theta,psi = x_nd[3:6]; v_nd = x_nd[6:9]; omega_nd_body = x_nd[9:12]
    R = R_from_euler_zyx(phi, theta, psi); q = quat_from_R(R); omega_nd_world = R @ omega_nd_body
    return np.concatenate([r,q]), np.concatenate([omega_nd_world, v_nd])


def mujoco_to_newton_x_from_nd(qpos_nd, qvel_nd, nd):
    r_nd = qpos_nd[0:3]; q = qpos_nd[3:7]; R = R_from_quat(q); phi,theta,psi = euler_zyx_from_R(R)
    omega_nd_world = qvel_nd[0:3]; v_nd = qvel_nd[3:6]; omega_nd_body = R.T @ omega_nd_world
    x_nd = np.concatenate([r_nd,[phi,theta,psi], v_nd, omega_nd_body])
    return x_nd_to_phys(x_nd, nd)


# ----------------- ND scales -----------------
def make_nd_scales(param):
    """
    Build nondimensional scales from current initial state and constants.
    L0 = ||r0||, T0 = sqrt(L0^3/mu), M0 = spacecraft mass.
    """
    mu = param["G"] * param["mass_earth"]
    L0 = float(np.linalg.norm(param["x0"][:3]))
    T0 = np.sqrt(L0**3 / mu)
    M0 = float(param["mass"])
    V0 = L0 / T0
    F0 = M0 * L0 / T0**2
    TAU0 = M0 * L0**2 / T0**2
    return dict(L0=L0, T0=T0, M0=M0, V0=V0, F0=F0, TAU0=TAU0)

# ----------------- ND conversions -----------------
def x_phys_to_nd(x, nd):
    r = x[0:3]/nd["L0"]
    phi,theta,psi = x[3:6]
    v = x[6:9]/nd["V0"]
    omega = x[9:12]*nd["T0"]
    return np.concatenate([r,[phi,theta,psi], v, omega])

def x_nd_to_phys(xn, nd):
    r = xn[0:3]*nd["L0"]
    phi,theta,psi = xn[3:6]
    v = xn[6:9]*nd["V0"]
    omega = xn[9:12]/nd["T0"]
    return np.concatenate([r,[phi,theta,psi], v, omega])

def u_to_nd(u, nd):
    f = u[0:3]/nd["F0"]
    tau = u[3:6]/nd["TAU0"]
    return np.concatenate([f,tau])

def inertia_phys_to_nd(I, nd):
    return I/(nd["M0"]*nd["L0"]**2)

# ----------------- ND gravity (force in ND units) -----------------
def force_gravity_world_nd(r_nd):
    R = np.linalg.norm(r_nd)
    if R < 1e-12: return np.zeros(3)
    return - r_nd / (R**3)  # since mu~ = 1 and m~ = 1