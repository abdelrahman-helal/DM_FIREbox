import torch
import numpy as np
from math import exp
import torch_geometric
from scipy.spatial import cKDTree
from numpy.random import default_rng
from sklearn.neighbors import NearestNeighbors


rng = default_rng(42)

def _pairwise_distance(a, b):
    """
    Euclidean distances from one 3-vector to many 3-vectors.

    Parameters:
    -----------
    a : numpy.ndarray
        Shape ``(3,)`` reference point.
    b : numpy.ndarray
        Shape ``(N, 3)`` collection of points.

    Returns:
    --------
    numpy.ndarray
        Length-``N`` distances.
    """
    return np.linalg.norm(b - a, axis=1)

def overlap_fraction(d, ri, rj):
    """
    Symmetric overlap fraction for two spheres given center separation and radii.

    Parameters:
    -----------
    d : float
        Distance between sphere centers.
    ri, rj : float
        Radii of the two spheres.

    Returns:
    --------
    float
        Value in ``[0, 1]``; zero when spheres do not overlap.
    """
    ov = max(0.0, (ri + rj - d) / (ri + rj))
    return ov

def build_spatial_ws_edges(pos_arr, radii, K, beta=0.1, lambda_decay=1.0, random_shortcuts=False):
    """
    Build a spatial Watts–Strogatz graph starting from kNN connectivity.

    Parameters:
    -----------
    pos_arr : numpy.ndarray
        Shape ``(N, 3)`` scaled coordinates.
    radii : numpy.ndarray
        Shape ``(N,)`` halo radii in the same distance units as ``pos_arr``.
    K : int
        Number of nearest neighbors (excluding self) before rewiring.
    beta : float
        Probability of rewiring each directed stub.
    lambda_decay : float
        Spatial kernel scale in normalized distance.
    random_shortcuts : bool
        If True, new endpoints are uniform random; else spatially weighted.

    Returns:
    --------
    edge_index : torch.Tensor
        Shape ``[2, E]`` undirected unique edges.
    edge_attr : dict
        Tensors with keys ``distance``, ``overlap``, and ``weight`` (column vectors).
    """
    N = pos_arr.shape[0]
    # initial kNN (symmetric)
    nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='kd_tree').fit(pos_arr)
    distances, indices = nbrs.kneighbors(pos_arr)
    # build adjacency set (directed list then symmetrize later)
    edges = set()
    for i, neigh in enumerate(indices):
        for j in neigh[1:]:
            edges.add((i, j))
            edges.add((j, i))

    # Convert to list for rewiring iteration
    edges = list(edges)
    # Rewire directed edges (i,j) by replacing j with k sampled appropriately
    for (i, j) in list(edges):  
        if rng.random() < beta:
            try:
                edges.remove((i, j))
            except ValueError:
                pass

            if random_shortcuts:
                candidates = np.arange(N)
                candidates = candidates[candidates != i]
                k = rng.choice(candidates)
            else:
                # spatial sampling proportionally to kernel
                dists = _pairwise_distance(pos_arr[i], pos_arr)  # (N,)
                # normalized distance s = d / (ri + rk)
                denom = (radii[i] + radii)
                denom = np.maximum(denom, 1e-8)
                s = dists / denom
                # kernel
                probs = np.exp(-s / lambda_decay)
                probs[i] = 0.0
                if probs.sum() <= 0:
                    # fallback to uniform
                    k = rng.choice(np.delete(np.arange(N), i))
                else:
                    probs = probs / probs.sum()
                    k = rng.choice(np.arange(N), p=probs)

            edges.append((i, k))

    # symmetrize and remove self-loops / duplicates
    edge_set = set()
    for (i, j) in edges:
        if i == j:
            continue
        a, b = min(i, j), max(i, j)
        edge_set.add((a, b))

    edge_list = np.array(list(edge_set), dtype=np.int64)
    if edge_list.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list.T, dtype=torch.long)

    # build edge attributes
    if edge_list.size == 0:
        edge_attr = {}
    else:
        dists = np.linalg.norm(pos_arr[edge_list[:, 0]] - pos_arr[edge_list[:, 1]], axis=1)
        overlap = np.array([overlap_fraction(d, radii[i], radii[j]) for d, i, j in zip(dists, edge_list[:,0], edge_list[:,1])])
        # weight by spatial kernel 
        weight = np.exp(- (dists / (radii[edge_list[:,0]] + radii[edge_list[:,1]] + 1e-8)) / lambda_decay)
        edge_attr = {
            'distance': torch.tensor(dists, dtype=torch.float).unsqueeze(1),
            'overlap': torch.tensor(overlap, dtype=torch.float).unsqueeze(1),
            'weight': torch.tensor(weight, dtype=torch.float).unsqueeze(1)
        }

    return edge_index, edge_attr


def build_spatial_ba_edges(pos_arr, radii, mass_array=None, m=2, a=0.01, lambda_decay=1.0, gamma=1.0,
                           order_by='mass', candidate_cutoff=None):
    """
    Grow a spatial Barabási–Albert graph with preferential attachment modulated by distance and overlap.

    Parameters:
    -----------
    pos_arr : numpy.ndarray
        Shape ``(N, 3)`` coordinates.
    radii : numpy.ndarray
        Shape ``(N,)`` radii for overlap and distance scaling.
    mass_array : numpy.ndarray or None
        Optional ``(N,)`` masses for arrival order when ``order_by='mass'``.
    m : int
        Target number of edges per arriving node.
    a : float
        Additive smoothing on degrees in attachment weights.
    lambda_decay : float
        Spatial decay constant in the attachment kernel.
    gamma : float
        Strength multiplier for halo-overlap boosting.
    order_by : str
        ``'mass'`` (requires ``mass_array``) or ``'random'`` arrival order.
    candidate_cutoff : float or None
        Optional max distance to existing nodes when sampling attachments.

    Returns:
    --------
    edge_index : torch.Tensor
        Undirected edges ``[2, E]``.
    edge_attr : dict
        Same structure as ``build_spatial_ws_edges`` when edges exist; empty dict if no edges.
    """
    N = pos_arr.shape[0]
    if order_by == 'mass' and mass_array is not None:
        # higher mass arrives earlier 
        order = np.argsort(-mass_array)
    else:
        order = np.arange(N)
        rng.shuffle(order)

    # start with small clique of size m (or m+1)
    m0 = max(m, 2)
    initial = list(order[:m0])
    # adjacency as set of undirected edges
    edge_set = set()
    degrees = np.zeros(N, dtype=np.int64)
    # fully connect initial clique
    for i in range(len(initial)):
        for j in range(i+1, len(initial)):
            a_i, a_j = initial[i], initial[j]
            edge_set.add((min(a_i,a_j), max(a_i,a_j)))
            degrees[a_i] += 1
            degrees[a_j] += 1

    # process remaining nodes in arrival order
    for t in order[m0:]:
        # potential existing nodes = nodes currently in graph
        existing = np.array([n for n in range(N) if n in order[:np.where(order==t)[0][0]]]) if False else np.array([n for n in range(N) if degrees[n] > 0])
        # fallback if no existing 
        if existing.size == 0:
            existing = np.array(initial)

        # candidate filtering by cutoff distance (if provided)
        if candidate_cutoff is not None:
            dists = np.linalg.norm(pos_arr[existing] - pos_arr[t], axis=1)
            mask = dists <= candidate_cutoff
            candidates = existing[mask]
            if candidates.size == 0:
                candidates = existing
        else:
            candidates = existing

        # compute attachment probabilities for candidates
        degs = degrees[candidates]
        dists = np.linalg.norm(pos_arr[candidates] - pos_arr[t], axis=1)
        denom = (radii[t] + radii[candidates])
        denom = np.maximum(denom, 1e-8)
        s = dists / denom
        spatial = np.exp(- s / lambda_decay)
        overlaps = np.maximum(0.0, (radii[t] + radii[candidates] - dists) / (radii[t] + radii[candidates] + 1e-8))
        fitness = (degs + a) * spatial * (1.0 + gamma * overlaps)
        # if all zeros, use uniform
        if fitness.sum() <= 0:
            probs = np.ones_like(fitness) / len(fitness)
        else:
            probs = fitness / fitness.sum()

        # choose up to m unique attachments
        chosen = set()
        # avoid self
        candidate_indices = candidates.copy()
        # sampling without replacement:
        k = min(m, len(candidate_indices))
        if len(candidate_indices) <= k:
            chosen_nodes = candidate_indices
        else:
            chosen_nodes = rng.choice(candidate_indices, size=k, replace=False, p=probs)

        for j in chosen_nodes:
            edge_set.add((min(t, j), max(t, j)))
            degrees[t] += 1
            degrees[j] += 1

    # build edge_list and edge_attr similar to WS
    edge_list = np.array(list(edge_set), dtype=np.int64)
    if edge_list.size == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = {}
    else:
        edge_index = torch.tensor(edge_list.T, dtype=torch.long)
        dists = np.linalg.norm(pos_arr[edge_list[:,0]] - pos_arr[edge_list[:,1]], axis=1)
        overlap = np.array([overlap_fraction(d, radii[i], radii[j]) for d, i, j in zip(dists, edge_list[:,0], edge_list[:,1])])
        weight = np.exp(- (dists / (radii[edge_list[:,0]] + radii[edge_list[:,1]] + 1e-8)) / lambda_decay)
        edge_attr = {
            'distance': torch.tensor(dists, dtype=torch.float).unsqueeze(1),
            'overlap': torch.tensor(overlap, dtype=torch.float).unsqueeze(1),
            'weight': torch.tensor(weight, dtype=torch.float).unsqueeze(1)
        }

    return edge_index, edge_attr
