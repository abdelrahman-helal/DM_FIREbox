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

    def create_graph_data(self, k=None, r=1, leaf_size=40, test_size=0.2, 
                        random_state=42, include_lg_Mstar=True, stratify_bins=10):
        """
        Create a PyTorch Geometric graph from the FIREbox data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The processed FIREbox dataframe
        k : int
            Number of nearest neighbors for graph connectivity
        test_size : float
            Fraction of data to use for testing
        random_state : int
            Random seed for reproducibility
        include_lg_Mstar : bool
            Whether to include lg_Mstar_<Rhalo in the feature set
        """
        
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
        
        # num_bins = 10
        # bins = np.linspace(y.min(), y.max(), num_bins)
        # y_binned = np.digitize(y, bins)
        # # Split data
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        # )
        
        # Create k-NN graph connectivity based on spatial coordinates
        pos_arr = X_scaled[:, 1:4]  # already scaled; KDTree on scaled coords is fine
        if k:
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(pos_arr)
            distances, indices = nbrs.kneighbors(pos_arr)   
            
            edges = []
            for i, neigh in enumerate(indices):
                for j in neigh[1:]:           # skip self
                    edges.append([i, j])
                    edges.append([j, i])      # add reverse for undirected behavior
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
            edge_index = torch_geometric.utils.to_undirected(edge_index)

        else:
            nbrs = cKDTree(pos_arr)
            indices = nbrs.query_pairs(r=r, output_type='ndarray')

            edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()  # [2, E]
            edge_index = torch_geometric.utils.to_undirected(edge_index)

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

        self.data = Data(
            x=torch.tensor(X_scaled, dtype=torch.float),       # node features
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.float).unsqueeze(1),
            pos=torch.tensor(pos_arr, dtype=torch.float),
        )
        self.data.train_mask = train_mask
        self.data.test_mask = test_mask

        return self.data


# if __name__ == "__main__":
#     data_processor = DataProcessor()
#     data_processor.create_graph_data()
#     print(data_processor.data)
#     print(data_processor.data.train_mask)
