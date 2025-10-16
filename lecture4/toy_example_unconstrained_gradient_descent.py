
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def algo_unconstrained_gradient_descent(x0, f, nablaf, m, M, lr="contraction"): 
	if lr == "contraction":
		alpha = 2/(M+m) 
		rho = (M - m)/(M + m)
	elif lr == "classical":
		alpha = 1/M 
		rho = 1 - m/M
	xs = [x0]
	for k in range(1000):
		grad = nablaf(xs[-1])
		xs.append(xs[-1] - alpha * grad)
		if np.linalg.norm(grad) < 1e-6:
			break
	fxs = [f(x) for x in xs]
	return xs, fxs, rho


def plot_one_experiment(xs, fxs, rho, title):
	fig, ax = plt.subplots()
	ax.plot(fxs - fxs[-1])
	# ax.plot(fxs)
	ax.plot([fxs[0] * rho**k for k in range(len(fxs))], '--')
	ax.set_title(title)
	ax.set_xlabel("Iteration")
	ax.set_ylabel("$f(x_k) - f^*$")
	ax.grid()


def plot_stats(ks):
	fig, ax = plt.subplots()
	ks_np = np.array(ks)
	seaborn.violinplot(data=ks_np, alpha=0.5)
	ax.set_xticks([0, 1], ['contraction', 'classical'])
	ax.set_ylabel("num timesteps before convergence")


if __name__ == '__main__':
	num_trials = 100
	dim = 10
	epsilon = 0.005
	ks = []
	for trial in range(num_trials):
		x0 = np.random.randn(dim)
		A = np.random.randn(dim, dim)
		A = A.T @ A + epsilon * np.eye(dim)  # make A positive definite
		b = np.random.randn(dim)
		c = np.random.randn()
		f = lambda x: 1 / 2 * x.T @ A @ x + b.T @ x + c
		nablaf = lambda x: A @ x + b
		m = np.linalg.eigvalsh(A).min()
		M = np.linalg.eigvalsh(A).max()
		xs_1, fxs_1, rho_1 = algo_unconstrained_gradient_descent(x0, f, nablaf, m, M, lr="contraction")
		xs_2, fxs_2, rho_2 = algo_unconstrained_gradient_descent(x0, f, nablaf, m, M, lr="classical")
		ks.append((len(xs_1), len(xs_2)))
		if num_trials == 1:
			plot_one_experiment(xs_1, fxs_1, rho_1, "contraction")
			plot_one_experiment(xs_2, fxs_2, rho_2, "classical")
	if num_trials > 1: 
		plot_stats(ks)
	plt.show()
