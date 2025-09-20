import torch
import numpy as np
import pandas as pd
import torch_geometric
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor():
    def __init__(self):
        self.df = None
        self.train_graph = None
        self.test_graph = None
        self.df_filtered = self.load_data()

    def load_data(self, file_path='data/FIREbox_z=0.txt'):
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
            - Position values divided by 10^2 to convert from dividing by H0/100 to H0
        """
        # Read the data file, skipping comment lines (starting with #)
        self.df = pd.read_csv(file_path, sep='\s+', comment='#')
        
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
        df_filtered['pos_x'] = df_filtered['pos_x'] / 100
        df_filtered['pos_y'] = df_filtered['pos_y'] / 100
        df_filtered['pos_z'] = df_filtered['pos_z'] / 100
        
        return df_filtered

    def create_graph_data(self, k=5, test_size=0.2, 
                        random_state=42, include_lg_Mstar=False):
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
        
        # Extract features and target
        X = self.df_filtered[feature_cols].values
        y = self.df_filtered[target_col].values
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Create k-NN graph connectivity based on spatial coordinates
        # Use pos_x, pos_y, pos_z (columns 1, 2, 3 in feature array)    
        
        # For training graph
        nbrs_train = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(X_train[:, 1:4])  
        distances_train, indices_train = nbrs_train.kneighbors(X_train[:, 1:4])
        pos_train = torch.tensor(X_train[:, 1:4], dtype=torch.float)

        # Remove self-loops (first neighbor is always the point itself)
        edge_index_train = []
        for i, neighbors in enumerate(indices_train):
            for j in neighbors[1:]:  # Skip first neighbor (self)
                edge_index_train.append([i, j])
        
        edge_index_train = torch.tensor(edge_index_train, dtype=torch.long).t().contiguous()
        
        # For test graph
        nbrs_test = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(X_test[:, 1:4])
        distances_test, indices_test = nbrs_test.kneighbors(X_test[:, 1:4])
        pos_test = torch.tensor(X_test[:, 1:4], dtype=torch.float)

        edge_index_test = []
        for i, neighbors in enumerate(indices_test):
            for j in neighbors[1:]:  # Skip first neighbor (self)
                edge_index_test.append([i, j])
        
        edge_index_test = torch.tensor(edge_index_test, dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data objects
        self.train_graph = Data(
            x=torch.tensor(X_train, dtype=torch.float),
            edge_index=edge_index_train,
            y=torch.tensor(y_train, dtype=torch.float).unsqueeze(1),
            pos=pos_train
        )
        
        self.test_graph = Data(
            x=torch.tensor(X_test, dtype=torch.float),
            edge_index=edge_index_test,
            y=torch.tensor(y_test, dtype=torch.float).unsqueeze(1), 
            pos=pos_test
        )
        
        return self.train_graph, self.test_graph


# if __name__ == "__main__":
#     data_processor = DataProcessor()
#     data_processor.create_graph_data()
#     print(data_processor.train_graph)
#     print(data_processor.test_graph)
