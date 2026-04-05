import torch
import numpy as np
import pandas as pd

import torch_geometric
from torch_geometric.data import Data

from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor():
    """
    Load FIREbox host halos and build a standard PyG graph (kNN or radius) for halo-mass regression.
    """

    def __init__(self, file_path='../data/FIREbox_z=0.txt'):
        """
        Initialize paths and load the filtered host-halo table.

        Parameters:
        -----------
        file_path : str
            Path to the FIREbox text catalog (whitespace-separated, ``#`` comments).

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
        Load FIREbox data from text file and process it.

        Parameters:
        -----------
        None
            Reads from ``self.file_path``.

        Returns:
        --------
        pd.DataFrame
            Processed dataframe with:
            - Only rows where hostHaloID = -1
            - Xc, Yc, Zc columns renamed to pos_x, pos_y, pos_z
            Also assigns the full table to ``self.df``.
        """
        # Read the data file, skipping comment lines (starting with #)
        self.df = pd.read_csv(self.file_path, sep=r'\s+', comment='#')
        
        # Filter for rows where hostHaloID = -1
        df_filtered = self.df[self.df['hostHaloID'] == -1].copy()
        
        # Rename position and velocity columns 
        df_filtered = df_filtered.rename(columns={
            'Xc': 'pos_x',
            'Yc': 'pos_y', 
            'Zc': 'pos_z', 
            'VXc': 'vel_x',
            'VYc': 'vel_y',
            'VZc': 'vel_z'
        })
        
        # Divide position values by 10^3
        # df_filtered['pos_x'] = df_filtered['pos_x'] / 100
        # df_filtered['pos_y'] = df_filtered['pos_y'] / 100
        # df_filtered['pos_z'] = df_filtered['pos_z'] / 100
        
        return df_filtered

    def create_graph_data(self, k=None, r=1, leaf_size=40, test_size=0.1, val_size=0.1,
                        standardize=True, random_state=42, include_Rhalo=False,
                        stratify_bins=10):
        """
        Filter galaxies, stratify train/val/test, optionally add Rhalo, and build kNN or radius edges.

        Parameters:
        -----------
        k : int or None
            Number of nearest neighbors (excluding self); if None, use radius graph with ``r``.
        r : float
            Pairwise distance threshold for radius graph when ``k`` is None.
        leaf_size : int
            Unused (reserved for tree backends); kept for API compatibility.
        test_size : float
            Fraction of nodes in the held-out test split.
        val_size : float
            Fraction of nodes in the validation split (relative to full graph).
        standardize : bool
            If True, fit ``StandardScaler`` on training rows only, then transform all rows.
        random_state : int
            Seed for stratified train/val/test splits.
        include_Rhalo : bool
            If True, append ``Rhalo`` to node features.
        stratify_bins : int or None
            Number of halo-mass percentile bins for stratified splitting; None disables stratification.

        Returns:
        --------
        torch_geometric.data.Data
            Graph with ``x``, ``edge_index``, ``y``, ``pos``, and ``train_mask`` / ``val_mask`` / ``test_mask``.
        """

        # Define feature columns and target
        self.df_filtered = self.df_filtered[self.df_filtered['lg_Mstar_<Rhalo'] > 0]

        feature_cols = ['pos_x', 'pos_y', 'pos_z',
                        'vel_x', 'vel_y', 'vel_z', 'lg_Mstar_<Rhalo']
        if include_Rhalo:
            feature_cols += ['Rhalo']

        target_col = 'lg_Mhalo'

        # Halo mass cutoff for galaxy formation
        self.df_filtered = self.df_filtered[self.df_filtered['lg_Mhalo'] > 9.5]

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
        # val_size fraction of the full dataset → fraction of temp set
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


# if __name__ == "__main__":
#     data_processor = DataProcessor()
#     data_processor.create_graph_data()
#     print(data_processor.data)
#     print(data_processor.data.train_mask)
