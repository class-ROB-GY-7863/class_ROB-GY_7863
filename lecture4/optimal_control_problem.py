
import numpy as np
import cvxpy as cp

def solve_trajectory(
    A, B, x0, N,
    Q=None, R=None, Qf=None,
    x_ref=None,
    x_min=None, x_max=None,
    u_min=None, u_max=None,
    x_goal=None,                   # if provided, enforces x_N == x_goal
    lambda_du=0.0,                 # penalty on control smoothness (Δu)
    solver="OSQP",                 # good default for QPs
    verbose=False
):
    """
    Solve a convex trajectory optimization for linear dynamics x_{t+1}=A x_t + B u_t.

    Args
    ----
    A (n x n), B (n x m): system matrices
    x0 (n,): initial state
    N (int): horizon length
    Q, R, Qf: PSD weight matrices (n x n, m x m, n x n). If None, defaults provided.
    x_ref: (N+1, n) reference states; if None, zeros are used.
    x_min, x_max: (n,) lower/upper bounds for state (optional, per-dimension box)
    u_min, u_max: (m,) lower/upper bounds for input (optional)
    x_goal: (n,) terminal equality (optional)
    lambda_du: nonnegative weight on sum ||u_{t+1}-u_t||_2^2
    solver: any cvxpy-supported QP solver ("OSQP", "ECOS", "GUROBI", etc.)
    verbose: cvxpy solver verbosity

    Returns
    -------
    dict with:
      "status": solver status
      "X": (N+1, n) optimal states
      "U": (N, m) optimal inputs
      "objective": optimal objective value
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    x0 = np.asarray(x0).reshape(-1)

    n = A.shape[0]
    m = B.shape[1]

    # defaults
    if Q is None:  Q = np.eye(n)
    if R is None:  R = 1e-2 * np.eye(m)
    if Qf is None: Qf = Q.copy()

    Q  = np.atleast_2d(Q)
    R  = np.atleast_2d(R)
    Qf = np.atleast_2d(Qf)

    if x_ref is None:
        x_ref = np.zeros((N+1, n))
    else:
        x_ref = np.asarray(x_ref)
        assert x_ref.shape == (N+1, n), "x_ref must be shape (N+1, n)"

    # decision variables
    X = cp.Variable((n, N+1))
    U = cp.Variable((m, N))

    constraints = []

    # initial condition
    constraints += [X[:, 0] == x0]

    # dynamics
    for t in range(N):
        constraints += [X[:, t+1] == A @ X[:, t] + B @ U[:, t]]

    # terminal equality (optional)
    if x_goal is not None:
        constraints += [X[:, N] == np.asarray(x_goal).reshape(-1)]

    # state bounds (optional)
    if x_min is not None:
        constraints += [X >= np.asarray(x_min).reshape(-1, 1)]
    if x_max is not None:
        constraints += [X <= np.asarray(x_max).reshape(-1, 1)]

    # input bounds (optional)
    if u_min is not None:
        constraints += [U >= np.asarray(u_min).reshape(-1, 1)]
    if u_max is not None:
        constraints += [U <= np.asarray(u_max).reshape(-1, 1)]

    # objective
    cost = 0
    for t in range(N):
        xt = X[:, t]
        ut = U[:, t]
        xr = x_ref[t, :]
        cost += cp.quad_form(xt - xr, Q) + cp.quad_form(ut, R)

    # terminal tracking
    cost += cp.quad_form(X[:, N] - x_ref[N, :], Qf)

    # optional smoothness on control: sum ||u_{t+1} - u_t||_2^2
    if lambda_du > 0 and N >= 2:
        for t in range(N - 1):
            cost += lambda_du * cp.sum_squares(U[:, t+1] - U[:, t])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver, verbose=verbose, warm_start=True)

    return {
        "status": prob.status,
        "objective": prob.value,
        "X": None if X.value is None else X.value.T,  # (N+1, n)
        "U": None if U.value is None else U.value.T,  # (N, m)
    }


# ------------------ Example usage: 2D point-mass double integrator ------------------
if __name__ == "__main__":
    dt = 0.1
    # state: [px, py, vx, vy], input: [ax, ay]
    A = np.block([
        [np.eye(2), dt*np.eye(2)],
        [np.zeros((2,2)), np.eye(2)]
    ])
    B = np.block([
        # [0.5*dt**2*np.eye(2)],
        [0*np.eye(2)],
        [dt*np.eye(2)]
    ])

    x0 = np.array([0, 0, 0, 0])
    x_goal = np.array([5, 3, 0, 0])

    N = 80

    # weights
    Q  = np.diag([2.0, 2.0, 0.1, 0.1])
    R  = 1e-2*np.eye(2)
    Qf = np.diag([10.0, 10.0, 1.0, 1.0])

    # straight-line reference in position (optional)
    x_ref = np.zeros((N+1, 4))
    x_ref[:, 0] = np.linspace(x0[0], x_goal[0], N+1)
    x_ref[:, 1] = np.linspace(x0[1], x_goal[1], N+1)

    # constraints
    x_min = np.array([-10, -10, -3, -3])
    x_max = np.array([ 10,  10,  3,  3])
    u_min = -1.5*np.ones(2)
    u_max =  1.5*np.ones(2)

    sol = solve_trajectory(
        A, B, x0, N,
        Q=Q, R=R, Qf=Qf,
        x_ref=x_ref,
        x_min=x_min, x_max=x_max,
        u_min=u_min, u_max=u_max,
        x_goal=x_goal,
        lambda_du=1e-2,
        solver="OSQP",
        verbose=False
    )

    print("Status:", sol["status"])
    print("Objective:", sol["objective"])
    print("X shape:", None if sol["X"] is None else sol["X"].shape)
    print("U shape:", None if sol["U"] is None else sol["U"].shape)
