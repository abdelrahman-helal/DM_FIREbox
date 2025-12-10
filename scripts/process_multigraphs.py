import torch
import numpy as np
import pandas as pd

import torch_geometric
from torch_geometric.data import Data

from form_graphs import build_spatial_ws_edges, build_spatial_ba_edges, overlap_fraction

from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor():
    def __init__(self, file_path='../data/FIREbox_z=0.txt'):
        self.file_path = file_path
        self.df = None
        self.data = Data()
        self.df_filtered = self.load_data()

    def load_data(self):
        """
        Load FIREbox data from text file and process it.
        
        Parameters:
        -----------
        file_path : str
            Path to the FIREbox_z=0.txt file
            
        Returns:
        --------
        pd.DataFrame
            Processed dataframe with:
            - Only rows where hostHaloID = -1
            - Xc, Yc, Zc columns renamed to pos_x, pos_y, pos_z
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
    
    def create_graph_data(self, k=None, r=1, leaf_size=40, test_size=0.2, standardize=True,
                      random_state=42, include_lg_Mstar=True, stratify_bins=10,
                      graph_type='knn',       # 'knn', 'ws', 'ba', or 'radius'
                      ws_beta=0.1, ws_lambda=0.5, ws_random_shortcuts=False,
                      ba_m=2, ba_a=0.01, ba_lambda=0.5, ba_gamma=1.0, ba_order_by='mass', ba_cutoff=None):

                # Define feature columns and target
        if include_lg_Mstar:
            feature_cols = ['Rhalo', 'pos_x', 'pos_y', 'pos_z',\
                            'vel_x', 'vel_y', 'vel_z', 'lg_Mstar_<Rhalo']
            self.df_filtered = self.df_filtered[self.df_filtered['lg_Mstar_<Rhalo'] > 0]
        else:
            feature_cols = ['Rhalo', 'pos_x', 'pos_y', 'pos_z', \
                            'vel_x', 'vel_y', 'vel_z']

        target_col = 'lg_Mhalo'
        
        # Halo mass cuttof for galaxy formation
        self.df_filtered = self.df_filtered[self.df_filtered['lg_Mhalo'] > 9.5]
        # Extract features and target
        X = self.df_filtered[feature_cols].values
        y = self.df_filtered[target_col].values
        
        # Normalize features
        if standardize:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        pos_arr = X_scaled[:, 1:4]  # scaled positions
        radii = X_scaled[:, 0]      # you used 'Rhalo' as first col; ensure Rhalo is in same units as pos scaling
        # NOTE: if Rhalo is not in the same scale (e.g., it's physical radius vs scaled coords),
        # you may prefer to use original radii (self.df_filtered['Rhalo'].values) instead of scaled.

        if graph_type == 'knn':
            # keep your existing kNN logic
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(pos_arr)
            distances, indices = nbrs.kneighbors(pos_arr)

            edges = []
            for i, neigh in enumerate(indices):
                for j in neigh[1:]:
                    edges.append([i, j])
                    edges.append([j, i])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = torch_geometric.utils.to_undirected(edge_index)
            # build simple edge_attr (distance, weight)
            # compute unique undirected edges:
            undirected = edge_index.t().numpy()
            unique_pairs = set()
            for a,b in undirected:
                if a==b: continue
                unique_pairs.add((min(a,b), max(a,b)))
            edge_list = np.array(list(unique_pairs), dtype=np.int64)
            if edge_list.size > 0:
                dists = np.linalg.norm(pos_arr[edge_list[:,0]] - pos_arr[edge_list[:,1]], axis=1)
                overlap = np.array([overlap_fraction(d, radii[i], radii[j]) for d,i,j in zip(dists, edge_list[:,0], edge_list[:,1])])
                weight = np.exp(- dists / ( (radii[edge_list[:,0]] + radii[edge_list[:,1]] + 1e-8) * ws_lambda))
                edge_attr = {
                    'distance': torch.tensor(dists, dtype=torch.float).unsqueeze(1),
                    'overlap': torch.tensor(overlap, dtype=torch.float).unsqueeze(1),
                    'weight': torch.tensor(weight, dtype=torch.float).unsqueeze(1)
                }
            else:
                edge_attr = {}

        elif graph_type == 'ws':
            # generate spatial WS using your helper
            edge_index, edge_attr = build_spatial_ws_edges(pos_arr, radii, K=k, beta=ws_beta, lambda_decay=ws_lambda,
                                                        random_shortcuts=ws_random_shortcuts)

        elif graph_type == 'ba':
            # use lg_Mhalo as mass ordering by default (you computed y earlier)
            mass_array = y  # y is lg_Mhalo array from above
            edge_index, edge_attr = build_spatial_ba_edges(pos_arr, radii, mass_array=mass_array,
                                                        m=ba_m, a=ba_a, lambda_decay=ba_lambda,
                                                        gamma=ba_gamma, order_by=ba_order_by,
                                                        candidate_cutoff=ba_cutoff)
        else:  # radius / FoF
            nbrs = cKDTree(pos_arr)
            indices = nbrs.query_pairs(r=r, output_type='ndarray')
            edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
            edge_index = torch_geometric.utils.to_undirected(edge_index)
            # compute edge_attr as above (omitted for brevity)
            edge_attr = {}

        N = len(self.df_filtered)
        idx = np.arange(N)

        # stratify by mass bins to keep mass distribution across train/test
        if stratify_bins is not None:
            bins = np.percentile(y, np.linspace(0, 100, stratify_bins + 1))
            y_bins = np.digitize(y, bins[1:-1])
            train_idx, test_idx = train_test_split(idx, test_size=test_size,
                                                random_state=random_state,
                                                stratify=y_bins)
        else:
            train_idx, test_idx = train_test_split(idx, test_size=test_size,
                                                random_state=random_state)
        
        train_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        # Attach edge attributes to PyG data object
        # PyG expects edge_attr aligned with edge_index columns; for simplicity we keep attributes per undirected edge list
        # If edge_attr exists as dict with tensors sized [E,1], we can combine them into single edge_attr tensor:
        if edge_attr:
            # stack attributes into one tensor [E, n_attr]
            attr_list = [edge_attr[k] for k in ['distance', 'overlap', 'weight'] if k in edge_attr]
            edge_attr_tensor = torch.cat(attr_list, dim=1)  # shape [E, num_attr]
            self.data = Data(
                x=torch.tensor(X_scaled, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr_tensor,
                y=torch.tensor(y, dtype=torch.float).unsqueeze(1),
                pos=torch.tensor(pos_arr, dtype=torch.float),
            )
        else:
            self.data = Data(
                x=torch.tensor(X_scaled, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.float).unsqueeze(1),
                pos=torch.tensor(pos_arr, dtype=torch.float),
            )

        # masks unchanged...
        self.data.train_mask = train_mask
        self.data.test_mask = test_mask

        return self.data
