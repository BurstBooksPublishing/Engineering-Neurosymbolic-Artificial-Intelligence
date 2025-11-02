import torch
from torch_geometric.data import Data

# Example features for 3 events, each with a 4-dimensional feature vector
node_features = torch.tensor([
    [0.5, 0.1, 0.3, 0.1],
    [0.1, 0.2, 0.6, 0.1],
    [0.3, 0.7, 0.1, 0.9]
], dtype=torch.float)

# Define connections between nodes: (source, target)
edge_index = torch.tensor([
    [0, 1],
    [1, 2],
    [2, 0]
], dtype=torch.long)

# Create a graph
graph = Data(x=node_features, edge_index=edge_index.t().contiguous())