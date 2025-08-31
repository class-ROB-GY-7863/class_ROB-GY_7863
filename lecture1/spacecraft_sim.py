
import numpy as np 
import h5py
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


# implement system from lecture 1

# x = [x,y,z,phi,theta,psi,vx,vy,vz,p,q,r]
# u = [fx,fy,fz,taux,tauy,tauz]


def simulate(fn):

    param = {
        "mass" : 1600, # kg 
        "radius" : 2.4, # m 
        "height" : 6.2, # m
        "mass_earth" : 5.97219e24, # kg
        "r_earth" : np.array([0,0,0]), # coordinate of earth 
        "radius_earth" : 6371e3, # m
        "G" : 6.67430e-11, 
    }

    # inertia estimated from cylinder formula: Iz = 1/2 m r^2, Ix = Iy = 1/12 m (3 r^2 + h^2)
    Ix = 1.0 / 12.0 * param["mass"] * (3 * param["radius"] ** 2.0 + param["height"] ** 2.0)
    Iz = 1.0 / 2.0 * param["mass"] * param["radius"] ** 2.0
    param["inertia"] = np.array([[Ix, 0, 0],[0, Ix, 0],[0, 0, Iz]]) # kg m2

    # initial state
    # x0 = np.array([param["radius_earth"] + 500e3, 0, 0, 0, 0, 0, 0, 7670, 0, 0, 0, 0]) # initial state
    x0 = np.array([param["radius_earth"] + 500e3, 0, 0, 0, 0, 0, 0, 7670, 1000, 0.001, 0, 0]) # initial state

    # empty controller 
    empty_controller = lambda x : np.array([0, 0, 0, 0, 0, 0])

    # simulate for 128 minutes of real time (LEO orbit)
    dt = 0.1
    num_steps = int(128 * 60 / dt)

    # simulate 
    xs = [x0]
    us = []
    for step in range(num_steps):
        print(f"step/total: {step}/{num_steps}")
        u = empty_controller(xs[-1])
        us.append(u)
        x_next = xs[-1] + dxdt(xs[-1], u, param) * dt
        xs.append(x_next)

    # save 
    save(fn, xs, param)


def dxdt(x, u, param): 
    # x is np array in (12,)
    # u is np array in (6,)

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
    _dxdt[3:6] = np.linalg.pinv(B(Theta)) @ omega 
    _dxdt[6:9] = f_g + f_ext 
    _dxdt[9:12] = np.linalg.pinv(param["inertia"]) @ (- np.cross(omega, param["inertia"] @ omega) + tau_ext)

    return _dxdt


def B(Theta):
    phi, theta, psi = Theta[0], Theta[1], Theta[2]
    return np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), - np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])


def save(fn, xs, params):
    with h5py.File(fn, "w") as h5file:
        h5file.create_dataset("xs", data=xs)
        for key, value in params.items():
            if isinstance(value, (int, float, str)):
                h5file.attrs[key] = value
            elif isinstance(value, (list, tuple)):
                h5file.attrs[key] = str(value)  # Convert lists/tuples to strings

def load(fn):
    with h5py.File(fn, "r") as h5file:
        xs = h5file["xs"][:]
        params = {key: h5file.attrs[key] for key in h5file.attrs}
    return xs, params


def render(fn):
    xs, params = load(fn)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color="blue")
    # downsample for body frame visualization
    stepsize = max(1, len(xs) // 20)
    for step in range(0, len(xs), stepsize):
        plot_body_frame(ax, xs[step, 0:3], xs[step, 3], xs[step, 4], xs[step, 5], scale=1e6)
    # Add a sphere for the Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = params["radius_earth"] * np.outer(np.cos(u), np.sin(v))
    y = params["radius_earth"] * np.outer(np.sin(u), np.sin(v))
    z = params["radius_earth"] * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="green", alpha=0.5)
    # Set labels and aspect
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("Spacecraft Trajectory with Earth")
    plt.show()


def euler_to_rotation_matrix(phi, theta, psi):
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


def plot_body_frame(ax, position, phi, theta, psi, scale=1.0):
    # Compute the rotation matrix
    orientation = euler_to_rotation_matrix(phi, theta, psi)
    # Extract the body frame axes
    x_body, y_body, z_body = orientation[:, 0], orientation[:, 1], orientation[:, 2]
    # Plot the body frame axes
    ax.quiver(
        position[0], position[1], position[2],  # Origin
        x_body[0], x_body[1], x_body[2],       # X-axis direction
        color="red", length=scale)
    ax.quiver(
        position[0], position[1], position[2],  # Origin
        y_body[0], y_body[1], y_body[2],       # Y-axis direction
        color="green", length=scale)
    ax.quiver(
        position[0], position[1], position[2],  # Origin
        z_body[0], z_body[1], z_body[2],       # Z-axis direction
        color="blue", length=scale)


if __name__ == '__main__':
    simulate("./spacecraft_sim.h5")
    render("./spacecraft_sim.h5")
