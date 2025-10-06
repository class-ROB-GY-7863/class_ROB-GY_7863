
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def algo_primal_dual(z0, A, b, c, D, g, max_num_iterations, lagrangian_mode="standard"): 

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
	for k in range(max_num_iterations-1):
		_dLdx = dLdx(zs[-1])
		_dLdl = dLdl(zs[-1])
		zs.append((
			zs[-1][0] - alpha * _dLdx,
			zs[-1][1] + beta * _dLdl
		))
		# there is a better way to break here
		if np.linalg.norm(_dLdx) < 1e-6:
			break
	fs = [f(z[0]) for z in zs]
	eq_constraints = [D @ z[0] + g for z in zs]
	rho = max(rho_x, rho_l)
	return zs, fs, eq_constraints, rho


def plot_one_experiment(zs, fs, eq_constraints, rho, label, fig=None, ax=None):
	if fig is None or ax is None:
		fig, ax = plt.subplots()
	ln = ax.plot(np.nan)
	color = ln[0].get_color()
	ax.plot(fs - fs[-1], color=color, label=label)
	ax.plot([fs[0] * rho**k for k in range(len(fs))], '--', color=color)
	ax.set_xlabel("Iteration")
	ax.set_ylabel("$f(x_k) - f^*$")
	ax.grid()
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
	
	num_trials = 100
	max_num_iterations = 1000
	dim_state = 10
	dim_constraint = 2
	epsilon = 0.5
	Fs = []
	
	for trial in range(num_trials):

		x0 = np.random.randn(dim_state)
		l0 = np.zeros(dim_constraint)

		A = np.random.randn(dim_state, dim_state)
		A = A.T @ A + epsilon * np.eye(dim_state) # make A positive definite
		b = np.random.randn(dim_state)
		c = np.random.randn()
		D = np.random.randn(dim_constraint, dim_state) 
		g = np.random.randn(dim_constraint) 

		zs_1, fs_1, eq_constraints_1, rho_1 = algo_primal_dual((x0, l0), A, b, c, D, g, max_num_iterations, lagrangian_mode="standard") 
		zs_2, fs_2, eq_constraints_2, rho_2 = algo_primal_dual((x0, l0), A, b, c, D, g, max_num_iterations, lagrangian_mode="augmented") 
		zs_3, fs_3, eq_constraints_3, rho_3 = algo_primal_dual((x0, l0), A, b, c, D, g, max_num_iterations, lagrangian_mode="regularized") 

		Fs.append((fs_1, fs_2, fs_3))

		if num_trials < 1000:
			fig, ax = plot_one_experiment(zs_1, fs_1, eq_constraints_1, rho_1, "standard")
			plot_one_experiment(zs_2, fs_2, eq_constraints_2, rho_2, "augmented", fig, ax)
			plot_one_experiment(zs_3, fs_3, eq_constraints_3, rho_3, "regularized", fig, ax)
			ax.legend()

	if num_trials > 1: 
		plot_stats(Fs, max_num_iterations)

	plt.show()
