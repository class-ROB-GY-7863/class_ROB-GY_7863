
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tqdm


def _pinv(A, rtol=1e-12):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = rtol * max(A.shape) * (s[0] if s.size else 0.0)
    s_inv = np.where(s > tol, 1.0/s, 0.0)
    return (Vt.T * s_inv) @ U.T


def _nullspace(A, rtol=1e-12):
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    tol = rtol * max(A.shape) * (s[0] if s.size else 0.0)
    r = np.sum(s > tol)
    return Vt[r:].T  # (n, n-r)


def _equality_min_norm(D, g, tol=1e-8):
    DDt = D @ D.T
    y = _pinv(DDt) @ g
    x = D.T @ y
    if np.linalg.norm(D @ x - g) > tol * (1 + np.linalg.norm(g)):
        raise ValueError("Equalities Dx=g appear inconsistent.")
    return x


def _strict_interior_on_affine(E, h, D, x_eq, max_outer=30, max_backtrack=25):
    N = _nullspace(D)
    x = x_eq.copy()
    if N.size == 0:
        if np.all(-(E @ x + h) > 0):
            return x
        raise ValueError("No strict interior within Dx=g (unique equality point lies on boundary).")
    EN = E @ N
    for _ in range(max_outer):
        e = E @ x + h
        svals = -e
        if np.all(svals > 0):
            return x
        tau = 0.1 * (1.0 + np.linalg.norm(e, ord=np.inf))
        dz = _pinv(EN) @ (-e - tau)
        p  = N @ dz
        t = 1.0
        smin_old = np.min(svals)
        for _ in range(max_backtrack):
            x_try = x + t * p
            s_try = -(E @ x_try + h)
            if np.min(s_try) > 0.0:
                return x_try
            if np.min(s_try) > smin_old:
                x = x_try
                break
            t *= 0.5
    raise ValueError("Failed to find strict interior by heuristic search.")


def analytic_center(D, g, E, h, max_inner_iterations):
    # Feasible, strictly interior start
    x0 = _equality_min_norm(D, g)
    x0 = _strict_interior_on_affine(E, h, D, x0)

    l0 = np.zeros(D.shape[0])

    f = lambda x: - np.sum(np.log(-(E @ x + h)))
    dfdx = lambda x: - E.T @ (1.0 / (E @ x + h))  # = E^T / (-(Ex+h))

    dLdx = lambda z: dfdx(z[0]) + D.T @ z[1]
    dLdl = lambda z: D @ z[0] + g

    # these can be selected better 
    alpha = 0.1
    beta  = 0.1

    zs = [(x0, l0)]
    for k in range(max_inner_iterations):
        _dLdx = dLdx(zs[-1])
        _dLdl = dLdl(zs[-1])

        # line search to keep Ex+h < 0
        p = 1.0
        while True:
            x_try = zs[-1][0] - p * alpha * _dLdx
            if np.all(-(E @ x_try + h) > 0):
                break
            p *= 0.5
            if p < 1e-12:
                raise RuntimeError("line search (in analytic center) failed to maintain log domain.")

        zs.append((
            x_try, 
            zs[-1][1] + beta * _dLdl
            ))

        if np.linalg.norm(_dLdx) < 1e-6 and np.linalg.norm(_dLdl) < 1e-6:
            break

    return zs[-1]


def algo_barrier(z0, A, b, c, D, g, E, h, t, mu0, max_outer_iterations, max_inner_iterations, lagrangian_mode="standard"): 

    f = lambda x, mu: 1 / mu * (1 / 2 * x.T @ A @ x + b.T @ x + c)
    dfdx = lambda x, mu: 1 / mu * (A @ x + b)

    _, sigmas_D, _ = np.linalg.svd(D)
    sigmasqrd_min = np.min(sigmas_D) ** 2.0
    sigmasqrd_max = np.max(sigmas_D) ** 2.0
    lambda_max_A, lambda_min_A = np.linalg.eigvalsh(A).max(), np.linalg.eigvalsh(A).min()

    # there are better ways to pick these constants
    gamma = 0.1
    eta = 0.1

    if lagrangian_mode == "standard":
        m = lambda_min_A
        M = lambda_max_A
        L = lambda z, mu: f(z[0], mu) + z[1].T @ (D @ z[0] + g) 
        dLdx = lambda z, mu: dfdx(z[0], mu) + D.T @ z[1] 
        dLdl = lambda z: D @ z[0] + g
    elif lagrangian_mode == "augmented":
        m = lambda_min_A + gamma * sigmasqrd_min
        M = lambda_max_A + gamma * sigmasqrd_max
        L = lambda z, mu: f(z[0], mu) + z[1].T @ (D @ z[0] + g) + gamma / 2 * np.linalg.norm(D @ z[0] + g)**2
        dLdx = lambda z, mu: dfdx(z[0], mu) + D.T @ z[1] + gamma * D.T @ (D @ z[0] + g) 
        dLdl = lambda z: D @ z[0] + g 
    elif lagrangian_mode == "regularized":
        # there are better ways to pick these constants
        m = lambda_min_A + gamma * sigmasqrd_min
        M = lambda_max_A + gamma * sigmasqrd_max
        L = lambda z, mu: f(z[0], mu) + z[1].T @ (D @ z[0] + g) + gamma / 2 * np.linalg.norm(D @ z[0] + g)**2 + eta/2 * np.linalg.norm(z[1])**2
        dLdx = lambda z, mu: dfdx(z[0], mu) + D.T @ z[1] + gamma * D.T @ (D @ z[0] + g) 
        dLdl = lambda z: D @ z[0] + g + eta * z[1]
    else:
        raise NotImplementedError

    alpha = 2/(M+m) 
    kappa_x = M/m 
    rho_x = (kappa_x - 1) / (kappa_x + 1)
    beta = 2/(sigmasqrd_min + sigmasqrd_max)
    kappa_l = sigmasqrd_max/sigmasqrd_min 
    rho_l = (kappa_l - 1) / (kappa_l + 1)

    zs = [z0]
    mus = [mu0]
    fs = [f(zs[-1][0], mus[-1])]
    for l in range(max_outer_iterations):

        phi = lambda x: - np.sum(np.log(-(E @ x + h))) 
        dphidx = lambda x: - E.T @ (1.0 / (E @ x + h)) 
        dLdx_with_barrier = lambda z, mu: dLdx(z, mu) + dphidx(z[0]) 

        for k in range(max_inner_iterations):

            _dLdx = dLdx_with_barrier(zs[-1], mus[-1])
            _dLdl = dLdl(zs[-1])

            x_try = zs[-1][0] - mus[-1] * alpha * _dLdx

            # # line search to keep Ex+h < 0
            # p = 1.0
            # while True:
            #     x_try = zs[-1][0] - p * mus[-1] * alpha * _dLdx
            #     if np.all(-(E @ x_try + h) > 0):
            #         break
            #     p *= 0.5
            #     if p < 1e-12:
            #         raise RuntimeError("line search failed to maintain log domain.")

            zs.append((
                x_try,
                zs[-1][1] + beta * _dLdl
            ))
            fs.append(f(zs[-1][0], mus[-1]))

            tol = 1e-6
            stationarity = np.linalg.norm(_dLdx)
            dual_resid   = np.linalg.norm(_dLdl)
            if stationarity < tol and dual_resid < tol:
                break
        
        # print("num inner iterations: ", k)
        mus.append(mus[-1] * t)
        # there is a better way to break here
        if mus[-1] <= 1e-3: 
            break 
    # print("num outer iterations: ", l)

    eq_constraints = [D @ z[0] + g for z in zs]
    ineq_constraints = [E @ z[0] + h for z in zs]
    rho = max(rho_x, rho_l)
    return zs, fs, eq_constraints, ineq_constraints, rho


def plot_one_experiment(zs, fs, eq_constraints, ineq_constraints, rho, label, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False)
    ln = ax[0,0].plot(np.nan)
    color = ln[0].get_color()
    ax[0,0].plot(np.abs(fs), color=color, label=label)
    # ax[0,0].plot([np.abs(fs[0]) * rho**k for k in range(len(fs))], '--', color=color)
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlabel("Iteration")
    # ax[0,0].set_ylabel("$f(x_k) - f^*$")
    ax[0,0].set_ylabel("$f(x_k)$")
    ax[0,0].grid()

    ax[0,1].plot([np.linalg.norm(c) for c in eq_constraints], color=color)
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel("Iteration")
    ax[0,1].set_ylabel("||Dx_k + g||")
    ax[0,1].grid()
    ax[0,2].plot([-np.min(c) for c in ineq_constraints], color=color)
    ax[0,2].set_yscale('log')
    ax[0,2].set_xlabel("Iteration")
    ax[0,2].set_ylabel("min(-Ex_k - h)")
    ax[0,2].grid()
    return fig, ax


def plot_stats(Fs, max_num_trials):

    Fs_np = np.nan*np.ones((len(Fs), 3, max_num_trials))
    ks = []
    for i, (f1, f2, f3) in enumerate(Fs):
        Fs_np[i, 0, :len(f1)] = f1
        Fs_np[i, 1, :len(f2)] = f2
        Fs_np[i, 2, :len(f3)] = f3
        ks.append((len(f1), len(f2), len(f3)))

    fig, ax = plt.subplots()
    ks_np = np.array(ks)
    seaborn.violinplot(data=ks_np, alpha=0.5)
    ax.set_xticks([0, 1, 2], ['standard', 'augmented', 'regularized'])
    ax.set_ylabel("num timesteps before convergence")


if __name__ == '__main__':
    
    seed = 0 
    num_trials = 10
    max_inner_iterations = 1000
    max_outer_iterations = 10
    dim_state = 10
    dim_eq_constraint = 2
    dim_ineq_constraint = 4
    epsilon = 0.5
    mu0 = 1.0 
    # t = 0.95
    t = 1.0
    Fs = []
    solvers = ["standard", "augmented", "regularized"]
    
    rng = np.random.default_rng(seed)

    for trial in tqdm.tqdm(range(num_trials)):

        A = rng.standard_normal((dim_state, dim_state))
        A = A.T @ A + epsilon * np.eye(dim_state) # make A positive definite
        b = rng.standard_normal((dim_state))
        c = rng.standard_normal(())
        D = rng.standard_normal((dim_eq_constraint, dim_state))
        g = rng.standard_normal((dim_eq_constraint))
        E = rng.standard_normal((dim_ineq_constraint, dim_state))
        h = rng.standard_normal((dim_ineq_constraint))

        try:
            x0, _ = analytic_center(D, g, E, h, max_inner_iterations)
        except: 
            print("analytic_center failed")
            continue
        l0 = np.zeros(dim_eq_constraint)

        z0 = (x0, l0)
        solns = []
        for solver in solvers:
            solns.append(algo_barrier(z0, A, b, c, D, g, E, h, t, mu0, max_outer_iterations, max_inner_iterations, lagrangian_mode=solver))

        Fs.append([soln[1] for soln in solns])

        if num_trials < 1000:
            for ii, (solver, soln) in enumerate(zip(solvers, solns)):
                if ii == 0:
                    fig, ax = plot_one_experiment(*soln, solver)
                else:
                    plot_one_experiment(*soln, solver, fig, ax)
            ax[0,0].legend()

    if num_trials > 1: 
        plot_stats(Fs, max_inner_iterations * max_outer_iterations + 1)

    plt.show()
