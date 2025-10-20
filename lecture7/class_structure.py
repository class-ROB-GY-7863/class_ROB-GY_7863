# pip install networkx matplotlib
# (optional for better layout) pip install pygraphviz  OR  pip install pydot graphviz

import networkx as nx
import matplotlib.pyplot as plt

USE_PYVIS = False

G = nx.DiGraph()
G.add_edges_from([
    # Top-level
    (
        "Reinforcement Learning \n Optimal Control",
        "Gradient-Based"
    ),
    (
        "Reinforcement Learning \n Optimal Control",
        "Search-Based"
    ),
    (
        "Reinforcement Learning \n Optimal Control",
        "Dynamic Programming"
    ),
    (
        "Reinforcement Learning \n Optimal Control",
        "Inference-Based"
    ),

    # Search-Based 
    (
        "Search-Based",
        "Tree-Based",
    ),
    (
        "Tree-Based",
        "RRT",
    ),
    (
        "Tree-Based",
        "FMT",
    ),
    (
        "Tree-Based",
        "MCTS",
    ),
    (
        "Search-Based",
        "Graph-Based",
    ),
    (
        "Graph-Based",
        "PRM",
    ),

    # Dynamic Programming
    (
        "Dynamic Programming",
        "Value Iteration"
    ),
    (
        "Dynamic Programming",
        "Policy Iteration"
    ),
    
    # Inference Based
    (
        "Inference-Based",
        "Bayesian"
    ),
    (
        "Inference-Based",
        "Frequentist"
    ),
    (
        "Bayesian", 
        "Kalman Filter"
    ),
    (
        "Bayesian", 
        "Planning as Inference"
    ),

    # Gradients
    (
        "Gradient-Based",
        r"Trajectory: $U$"
    ),
    (
        "Gradient-Based",
        r"Policy: $\pi: X \rightarrow U$"
    ),

    # Trajectory sub-branch
    (
        r"Trajectory: $U$",
        r"Zero Order: $J(U)$"
    ),
    (
        r"Trajectory: $U$",
        r"First Order: $\partial J / \partial U$"
    ),
    (
        r"Trajectory: $U$",
        r"Second Order: $\partial^2 J / \partial U^2$"
    ),
    (
        r"Trajectory: $U$",
        r"Affine Equality: $f_i(U)=0$"
    ),
    (
        r"Trajectory: $U$",
        r"Convex Inequality: $f_i(U)<0$"
    ),
    (
        r"Trajectory: $U$",
        r"Nonconvex Optimization",
    ),
    (
        r"First Order: $\partial J / \partial U$",
        "Gradient Descent"
    ),
    (
        r"Second Order: $\partial^2 J / \partial U^2$",
        "Newton Method"
    ),
    (
        r"Zero Order: $J(U)$",
        "Score Estimation"
    ),
    (
        r"Nonconvex Optimization",
        r"Sequential Convex Programming" 
    ),
    (
        r"Affine Equality: $f_i(U)=0$",
        "Lagrange Multipliers: \n" r"$Z=(U;\lambda)$"
    ),
    (
        r"Convex Inequality: $f_i(U)<0$",
        "Barrier Function: \n" r"$J_ρ = J - ρ \log(-g_i(U))$"
    ),
    (
        "Lagrange Multipliers: \n" r"$Z=(U;\lambda)$",
        r"First Order: $\partial J / \partial U$"
    ), 
    (
        "Barrier Function: \n" r"$J_ρ = J - ρ \log(-g_i(U))$",
        r"First Order: $\partial J / \partial U$"
    ), 
    (
        "Score Estimation", 
        r"First Order: $\partial J / \partial U$"
    ),
    (
        r"Sequential Convex Programming",
        r"First Order: $\partial J / \partial U$"
    ),

    # Policy 
    (
        r"Policy: $\pi: X \rightarrow U$",
        r"Linear Policy: $\pi(x) = -K x$"
    ),
    (
        r"Linear Policy: $\pi(x) = -K x$", 
        "Linear Dynamics, Quadratic Cost: \n"+r"$x_{k+1} = A x_k + B u_k$, $c_k = x_k^T Q x_k + u_k^T R u_k$"
    ),
    (
        "Linear Dynamics, Quadratic Cost: \n"+r"$x_{k+1} = A x_k + B u_k$, $c_k = x_k^T Q x_k + u_k^T R u_k$",
        "Linear Quadratic Regulator (DARE)"
    ),
    (
        r"Linear Policy: $\pi(x) = -K x$", 
        "Nonlinear Dynamics, Nonlinear Cost: \n"+r"$x_{k+1} = f(x_k, u_k)$, $c_k = g(x_k, u_k)$"
    ),
    (
        "Nonlinear Dynamics, Nonlinear Cost: \n"+r"$x_{k+1} = f(x_k, u_k)$, $c_k = g(x_k, u_k)$", 
        "First Order: iLQR"
    ),
    (
        "First Order: iLQR", 
        "Linear Dynamics, Quadratic Cost: \n"+r"$x_{k+1} = A x_k + B u_k$, $c_k = x_k^T Q x_k + u_k^T R u_k$",
    ),
    (
        "Nonlinear Dynamics, Nonlinear Cost: \n"+r"$x_{k+1} = f(x_k, u_k)$, $c_k = g(x_k, u_k)$", 
        "Second Order: DDP"
    ),
    (
        r"Policy: $\pi: X \rightarrow U$",
        r"Parameterized Policy: $\pi_\theta(x)$"
    ),

    # Reinforcement Learning 
    (
        r"Parameterized Policy: $\pi_\theta(x)$",
        "REINFORCE"
    ),
    (
        "REINFORCE",
        "PPO",
    ),
    (
        "REINFORCE",
        "SAC",
    ),
    (
        "PPO",
        "Gradient Descent",
    ),
    (
        "SAC",
        "Gradient Descent",
    ),

    # Connections
    (
        r"Score Estimation",
        r"REINFORCE"
    ),
    (
        r"REINFORCE",
        r"Score Estimation"
    ),
    (
        r"MCTS",
        r"Value Iteration"
    ),
    (
        r"FMT",
        r"Dynamic Programming"
    ),
    (
        "Linear Quadratic Regulator (DARE)",
        "Gradient Descent"
    ),
    (
        "Planning as Inference",
        "SAC"
    ),

    # Everything goes to contraction
    (
        "Gradient Descent",
        "Contraction"
    ),
    (
        "Newton Method",
        "Contraction"
    ), 
    (
        "Linear Quadratic Regulator (DARE)",
        "Contraction"
    ),
    (
        "Value Iteration",
        "Contraction"
    ),
    (
        "Policy Iteration",
        "Contraction"
    ),
])

# --- Layout: prefer Graphviz 'dot' layered layout when available ---
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

if not USE_PYVIS:
    # --- Draw ---
    # plt.figure(figsize=(12, 9))
    plt.figure(figsize=(24, 18))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1800,
        node_color="#E8F0FE",
        edge_color="#555555",
        linewidths=0.8,
        font_size=9
    )
    plt.tight_layout()
    plt.savefig("control_dag.png", dpi=220)
    plt.show()
else:
    # ---------- Interactive HTML via PyVis ----------
    try:
        from pyvis.network import Network
    except ImportError:
        raise SystemExit("PyVis not installed. Run: pip install pyvis jinja2")

    def html_label(s: str) -> str:
        """Convert LaTeX-ish bits to HTML-ish label; handle newlines & common symbols."""
        return (str(s)
                .replace("\n", "<br>")
                .replace("→", "&rarr;")
                .replace("∂", "&part;")
                .replace("π", "&pi;")
                .replace("θ", "&theta;")
                .replace("ρ", "&rho;")
                .replace("λ", "&lambda;"))

    net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", notebook=False)

    # Fix positions from Graphviz (so interactive matches the static look)
    net.toggle_physics(False)  # keep fixed layout (set True for force physics)

    for n in G.nodes():
        x, y = pos[n]
        net.add_node(
            n,
            label=html_label(n),
            title=f"<b>{html_label(n)}</b>",  # hover tooltip
            shape="box",
            x=float(x),
            y=float(-y),        # flip Y for screen coords
            fixed=False,
        )

    for u, v in G.edges():
        net.add_edge(u, v, arrows="to")

    # Avoid old template bug by forcing template load
    # net.set_template()
    net.show("control_dag.html", notebook=False)   # open in browser; creates control_dag.html
    # Alternatively:
    # net.write_html("control_dag.html", notebook=False)
