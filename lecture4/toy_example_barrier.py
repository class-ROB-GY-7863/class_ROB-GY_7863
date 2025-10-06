
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tqdm


def algo_barrier(z0, A, b, c, D, g, E, h, t, mu, max_outer_iterations, max_inner_iterations, lagrangian_mode="standard"): 

	f = lambda x: 1 / 2 * x.T @ A @ x + b.T @ x + c
	nablaf = lambda x: A @ x + b

	if lagrangian_mode == "standard":
		m = np.linalg.eigvalsh(A).min()
		M = np.linalg.eigvalsh(A).max()
		_, sigmas, _ = np.linalg.svd(D)
		sigmasqrd_min = np.min(sigmas) ** 2.0
		sigmasqrd_max = np.max(sigmas) ** 2.0
		L = lambda z: f(z[0]) + z[1].T @ (D @ z[0] + g) 
		dLdx = lambda z: nablaf(z[0]) + D.T @ z[1] 
		dLdl = lambda z: D @ z[0] + g
	elif lagrangian_mode == "augmented":
		# there are better ways to pick these constants
		gamma = 0.1 
		_, sigmas, _ = np.linalg.svd(D)
		sigmasqrd_min = np.min(sigmas) ** 2.0
		sigmasqrd_max = np.max(sigmas) ** 2.0
		m = np.linalg.eigvalsh(A).min() + gamma * sigmasqrd_min
		M = np.linalg.eigvalsh(A).max() + gamma * sigmasqrd_max
		L = lambda z: f(z[0]) + z[1].T @ (D @ z[0] + g) + gamma / 2 * np.linalg.norm(D @ z[0] + g)**2
		dLdx = lambda z: nablaf(z[0]) + D.T @ z[1] + gamma * D.T @ (D @ z[0] + g) 
		dLdl = lambda z: D @ z[0] + g 
	elif lagrangian_mode == "regularized":
		# there are better ways to pick these constants
		gamma = 0.1 
		eta = 0.1 
		_, sigmas, _ = np.linalg.svd(D)
		sigmasqrd_min = np.min(sigmas) ** 2.0
		sigmasqrd_max = np.max(sigmas) ** 2.0
		m = np.linalg.eigvalsh(A).min() + gamma * sigmasqrd_min
		M = np.linalg.eigvalsh(A).max() + gamma * sigmasqrd_max
		L = lambda z: f(z[0]) + z[1].T @ (D @ z[0] + g) + gamma / 2 * np.linalg.norm(D @ z[0] + g)**2 + eta/2 * np.linalg.norm(z[1])**2
		dLdx = lambda z: nablaf(z[0]) + D.T @ z[1] + gamma * D.T @ (D @ z[0] + g) 
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
	for l in range(max_outer_iterations):

		phi    = lambda x: -mu * np.sum(np.log(-(E @ x + h))) 
		dphidx = lambda x:  mu * E.T @ (1.0 / (E @ x + h)) 
		dLdx_with_barrier = lambda z: dLdx(z) + dphidx(z[0]) 

		for k in range(max_inner_iterations-1):

			_dLdx = dLdx_with_barrier(zs[-1])
			_dLdl = dLdl(zs[-1])

			zs.append((
				zs[-1][0] - alpha * _dLdx,
				zs[-1][1] + beta * _dLdl
			))

			# there is a better way to break here
			# if np.linalg.norm(_dLdx) < 1e-6:
			# 	break
			tol = 1e-6
			stationarity = np.linalg.norm(dLdx_with_barrier(zs[-1]))
			dual_resid   = np.linalg.norm(dLdl(zs[-1]))
			if stationarity < tol and dual_resid < tol:
				break
		
		# print("num inner iterations: ", k)
		mu *= t 
		# there is a better way to break here
		if mu <= 1e-3: 
			break 
	# print("num outer iterations: ", l)

	fs = [f(z[0]) for z in zs]
	eq_constraints = [D @ z[0] + g for z in zs]
	ineq_constraints = [E @ z[0] + h for z in zs]
	rho = max(rho_x, rho_l)
	return zs, fs, eq_constraints, ineq_constraints, rho


def plot_one_experiment(zs, fs, eq_constraints, ineq_constraints, rho, label, fig=None, ax=None):
	if fig is None or ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False)
	ln = ax[0,0].plot(np.nan)
	color = ln[0].get_color()
	ax[0,0].semilogy(fs - fs[-1], color=color, label=label)
	ax[0,0].semilogy([fs[0] * rho**k for k in range(len(fs))], '--', color=color)
	ax[0,0].set_xlabel("Iteration")
	ax[0,0].set_ylabel("$f(x_k) - f^*$")
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
	seaborn.violinplot(ks_np, alpha=0.5)
	ax.set_xticks([0, 1, 2], ['standard', 'augmented', 'regularized'])
	ax.set_ylabel("num timesteps before convergence")


if __name__ == '__main__':
	
	num_trials = 10
	max_outer_iterations = 1000
	max_inner_iterations = 1000
	dim_state = 10
	dim_eq_constraint = 2
	dim_ineq_constraint = 4
	epsilon = 0.5
	mu = 10.0 
	t = 0.75 
	Fs = []
	
	for trial in tqdm.tqdm(range(num_trials)):

		x0 = np.zeros(dim_state)
		l0 = np.zeros(dim_eq_constraint)

		A = np.random.randn(dim_state, dim_state)
		A = A.T @ A + epsilon * np.eye(dim_state) # make A positive definite
		b = np.random.randn(dim_state)
		c = np.random.randn()
		D = np.random.randn(dim_eq_constraint, dim_state) 
		g = np.random.randn(dim_eq_constraint) 
		E = np.random.randn(dim_ineq_constraint, dim_state) 
		h = np.random.randn(dim_ineq_constraint) 

		z0 = (x0, l0)
		zs_1, fs_1, eq_constraints_1, ineq_constraints_1, rho_1 = algo_barrier(z0, A, b, c, D, g, E, h, t, mu, max_outer_iterations, max_inner_iterations, lagrangian_mode="standard") 
		zs_2, fs_2, eq_constraints_2, ineq_constraints_2, rho_2 = algo_barrier(z0, A, b, c, D, g, E, h, t, mu, max_outer_iterations, max_inner_iterations, lagrangian_mode="augmented") 
		zs_3, fs_3, eq_constraints_3, ineq_constraints_3, rho_3 = algo_barrier(z0, A, b, c, D, g, E, h, t, mu, max_outer_iterations, max_inner_iterations, lagrangian_mode="regularized") 

		Fs.append((fs_1, fs_2, fs_3))

		if num_trials < 1000:
			fig, ax = plot_one_experiment(zs_1, fs_1, eq_constraints_1, ineq_constraints_1, rho_1, "standard")
			plot_one_experiment(zs_2, fs_2, eq_constraints_2, ineq_constraints_2, rho_2, "augmented", fig, ax)
			plot_one_experiment(zs_3, fs_3, eq_constraints_3, ineq_constraints_3, rho_3, "regularized", fig, ax)
			ax[0,0].legend()

	if num_trials > 1: 
		plot_stats(Fs, max_inner_iterations * max_outer_iterations)

	plt.show()
