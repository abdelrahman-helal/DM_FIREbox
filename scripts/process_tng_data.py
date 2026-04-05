import torch
import numpy as np
import pandas as pd

import torch_geometric
from torch_geometric.data import Data

from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TNGDataProcessor():
    """
    Load IllustrisTNG subhalo parquet and expose FIREbox-compatible graph construction.
    """

    def __init__(self, file_path='../data/tng-data/TNG300-1-subhalos_99.parquet'):
        """
        Initialize paths and load the renamed TNG subhalo table.

        Parameters:
        -----------
        file_path : str
            Path to the TNG subhalos parquet file.

        Returns:
        --------
        None
            Sets ``self.file_path``, ``self.df``, ``self.data``, and ``self.df_filtered``.
        """
        self.file_path = file_path
        self.df = None
        self.data = Data()
        self.df_filtered = self.load_data()

    def load_data(self):
        """
        Load TNG subhalos data from parquet file and process it.

        Parameters:
        -----------
        None
            Reads from ``self.file_path``.

        Returns:
        --------
        pd.DataFrame
            Processed dataframe with renamed columns to match FIREbox convention.
            Also assigns the raw table to ``self.df``.
        """
        # Read the parquet file
        self.df = pd.read_parquet(self.file_path)
        
        # Rename columns to match FIREbox naming convention
        df_filtered = self.df.rename(columns={
            'subhalo_x': 'pos_x',
            'subhalo_y': 'pos_y',
            'subhalo_z': 'pos_z',
            'subhalo_vx': 'vel_x',
            'subhalo_vy': 'vel_y',
            'subhalo_vz': 'vel_z',
            'subhalo_logstellarmass': 'stellar_mass'
        }).copy()
        
        return df_filtered

    def create_graph_data(self, k=None, r=1, leaf_size=40, test_size=0.1, val_size=0.1,
                        standardize=True, random_state=42, stratify_bins=10):
        """
        Create a PyTorch Geometric graph from the TNG data.

        Parameters:
        -----------
        k : int or None
            Number of nearest neighbors for graph connectivity (None → radius-based).
        r : float
            Radius for radius-based connectivity (used when ``k`` is None).
        leaf_size : int
            Unused; reserved for API compatibility.
        test_size : float
            Fraction of data held out as the final test set (default 0.10).
        val_size : float
            Fraction of data used for validation / early-stopping (default 0.10).
        standardize : bool
            Whether to standardize features (train-fit scaler).
        random_state : int
            Random seed for reproducibility.
        stratify_bins : int or None
            Number of mass-percentile bins used to stratify all splits.

        Returns:
        --------
        torch_geometric.data.Data
            Object with ``x``, ``edge_index``, ``y``, ``pos``, and train/val/test masks.
        """

        feature_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'stellar_mass']
        target_col = 'subhalo_loghalomass'

        self.df_filtered = self.df_filtered.dropna(subset=feature_cols + [target_col])

        X = self.df_filtered[feature_cols].values.astype(np.float64)
        y = self.df_filtered[target_col].values

        N = len(self.df_filtered)
        idx = np.arange(N)

        # ── 3-way stratified split (train / val / test) ──────────────────────
        # Split indices BEFORE fitting the scaler to prevent data leakage.
        if stratify_bins is not None:
            bins = np.percentile(y, np.linspace(0, 100, stratify_bins + 1))
            y_bins = np.digitize(y, bins[1:-1])
        else:
            y_bins = None

        temp_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state,
            stratify=y_bins)

        y_bins_temp = y_bins[temp_idx] if y_bins is not None else None
        val_frac_of_temp = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            temp_idx, test_size=val_frac_of_temp, random_state=random_state,
            stratify=y_bins_temp)

        # ── Fit scaler on training rows ONLY, then transform all rows ────────
        if standardize:
            self.scaler = StandardScaler()
            X[train_idx] = self.scaler.fit_transform(X[train_idx])
            X[val_idx]   = self.scaler.transform(X[val_idx])
            X[test_idx]  = self.scaler.transform(X[test_idx])
        X_scaled = X

        # ── Build graph on scaled positions (all nodes, transductive setting) ─
        pos_arr = X_scaled[:, 0:3]
        if k:
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(pos_arr)
            distances, indices = nbrs.kneighbors(pos_arr)

            edges = np.zeros((N * k * 2, 2), dtype=np.int64)
            edge_ptr = 0
            for i, neigh in enumerate(indices):
                for j in neigh[1:]:
                    edges[edge_ptr]     = [i, j]
                    edges[edge_ptr + 1] = [j, i]
                    edge_ptr += 2
            edges = edges[:edge_ptr]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = torch_geometric.utils.to_undirected(edge_index)
        else:
            nbrs = cKDTree(pos_arr)
            indices = nbrs.query_pairs(r=r, output_type='ndarray')
            edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
            edge_index = torch_geometric.utils.to_undirected(edge_index)

        # ── Build masks ───────────────────────────────────────────────────────
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask   = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        test_mask[test_idx]   = True

        self.data = Data(
            x=torch.tensor(X_scaled, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.float).unsqueeze(1),
            pos=torch.tensor(pos_arr, dtype=torch.float),
        )
        self.data.train_mask = train_mask
        self.data.val_mask   = val_mask
        self.data.test_mask  = test_mask

        return self.data

