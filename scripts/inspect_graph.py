import torch
import numpy as np
import seaborn as sns  
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.utils import to_networkx
import plotly.graph_objects as go


def pyg_to_nx(data):
    """
    Convert a PyG ``Data`` object to an undirected ``networkx.Graph`` with node attributes.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        Graph possibly containing ``x``, ``pos``, and ``y``.

    Returns:
    --------
    networkx.Graph
        Undirected graph with copied per-node features when present.
    """
    G = to_networkx(data, to_undirected=True)

    # Copy node attributes
    for n in G.nodes():
        if hasattr(data, "x"):
            G.nodes[n]["x"] = data.x[n].cpu().numpy()
        if hasattr(data, "pos"):
            G.nodes[n]["pos"] = data.pos[n].cpu().numpy()
        if hasattr(data, "y"):
            G.nodes[n]["y"] = float(data.y[n].cpu().item())
    return G


# Compute network diagnostics
def graph_metrics(G, topk_hubs=10):
    """
    Compute basic structural statistics and the highest-degree hubs.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.
    topk_hubs : int
        Number of top-degree nodes to list.

    Returns:
    --------
    dict
        Keys include ``N``, ``E``, degree summaries, ``avg_clustering``, and ``top_hubs``.
    """
    N = G.number_of_nodes()
    E = G.number_of_edges()

    degrees = np.array([d for _, d in G.degree()])
    deg_mean = degrees.mean()
    deg_var = degrees.var()
    deg_max = degrees.max()

    # clustering coefficient
    avg_clustering = nx.average_clustering(G)

    # hub nodes
    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:topk_hubs]

    return {
        "N": N,
        "E": E,
        "deg_mean": deg_mean,
        "deg_var": deg_var,
        "deg_max": deg_max,
        "avg_clustering": avg_clustering,
        "top_hubs": hubs
    }

# Degree CCDF plot
def plot_degree_ccdf(G, ax=None):
    """
    Plot the complementary CDF of the degree distribution on log-log axes.

    Parameters:
    -----------
    G : networkx.Graph
        Graph whose degrees are histogrammed.
    ax : matplotlib.axes.Axes or None
        Axis to draw on; if None, a new figure and axis are created.

    Returns:
    --------
    matplotlib.axes.Axes
        The axis containing the CCDF step plot.
    """
    degrees = np.array([d for _, d in G.degree()])
    vals, counts = np.unique(degrees, return_counts=True)

    cdf = np.cumsum(counts) / counts.sum()
    ccdf = 1 - cdf + counts / counts.sum()

    if ax is None:
        fig, ax = plt.subplots()

    ax.step(vals, ccdf, where="post")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree k")
    ax.set_ylabel("CCDF P(K ≥ k)")
    ax.set_title("Degree CCDF")
    return ax


# Edge-length distribution (needs node positions)
def plot_link_length_hist(G, bins=50, logy=True, ax=None):
    """
    Histogram Euclidean edge lengths using node ``pos`` attributes when present.

    Parameters:
    -----------
    G : networkx.Graph
        Graph with ``pos`` on nodes for length computation.
    bins : int
        Number of histogram bins.
    logy : bool
        If True, use a logarithmic y-axis.
    ax : matplotlib.axes.Axes or None
        Target axis; created if None.

    Returns:
    --------
    matplotlib.axes.Axes
        Axis with the histogram.
    """
    lengths = []
    for u, v in G.edges():
        pu = G.nodes[u].get("pos")
        pv = G.nodes[v].get("pos")
        if pu is not None and pv is not None:
            lengths.append(np.linalg.norm(pu - pv))

    lengths = np.array(lengths)

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(lengths, bins=bins)
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Edge length")
    ax.set_ylabel("Count")
    ax.set_title("Edge-length distribution")
    return ax


# 3D Visualization (subsample + limited edges)
def plot_3d_graph(G, graph_type='Graph',sample_frac=0.2, max_edges=20000, color_attr="y", node_size=5):
    """
    Render an interactive Plotly 3D scatter/line view of a subsampled graph.

    Parameters:
    -----------
    G : networkx.Graph
        Graph with ``pos`` attributes on nodes.
    graph_type : str
        Label used only in the figure title.
    sample_frac : float
        Fraction of nodes with positions to keep (random subsample).
    max_edges : int
        Cap on plotted edges after subsampling.
    color_attr : str or None
        Node attribute name for marker colors (default ``y`` / halo mass).
    node_size : float
        Plotly marker size.

    Returns:
    --------
    None
        Displays the figure via ``fig.show()``.
    """
    pos = {n: G.nodes[n].get("pos") for n in G.nodes()}
    nodes = [n for n, p in pos.items() if p is not None]

    # Subsample nodes
    if sample_frac < 1:
        import random
        nodes_s = set(random.sample(nodes, int(len(nodes) * sample_frac)))
    else:
        nodes_s = set(nodes)

    # Filter edges
    edges = [(u, v) for u, v in G.edges() if u in nodes_s and v in nodes_s]
    if len(edges) > max_edges:
        import random
        edges = random.sample(edges, max_edges)

    # Build arrays
    xs = np.array([pos[n][0] for n in nodes_s])
    ys = np.array([pos[n][1] for n in nodes_s])
    zs = np.array([pos[n][2] for n in nodes_s])

    if color_attr:
        colors = np.array([G.nodes[n].get(color_attr, 0.0) for n in nodes_s])
    else:
        colors = None

    # Create Plotly figure
    fig = go.Figure()

    # Add edges as line traces
    edge_x = []
    edge_y = []
    edge_z = []
    for u, v in edges:
        p1, p2 = pos[u], pos[v]
        edge_x.extend([p1[0], p2[0], None])
        edge_y.extend([p1[1], p2[1], None])
        edge_z.extend([p1[2], p2[2], None])
    
    fig.add_trace(go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='rgba(255, 150, 150, 0.05)', width=1),
        hoverinfo='skip',
        showlegend=False
    ))

    # Add nodes as scatter plot
    hover_text = [f"Node {n}<br>X: {xs[i]:.2f}<br>Y: {ys[i]:.2f}<br>Z: {zs[i]:.2f}" + 
                  (f"<br>{color_attr}: {colors[i]:.2f}" if color_attr else "")
                  for i, n in enumerate(nodes_s)]
    
    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(
            size=node_size,
            color=colors if color_attr else 'blue',
            colorscale='Viridis',
            colorbar=dict(title='Log Halo Mass') if color_attr else None,
            showscale=color_attr is not None,
            line=dict(width=0.5, color='rgba(0, 0, 0, 0.3)')
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Nodes'
    )
    fig.add_trace(node_trace)

    # Update layout
    fig.update_layout(
        title=f'3D {graph_type} Visualization',
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.show()
