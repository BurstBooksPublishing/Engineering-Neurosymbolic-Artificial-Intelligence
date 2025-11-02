import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Example graph: nodes represent entities and edges represent relationships
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# Create a graph
graph = Data(x=x, edge_index=edge_index.t().contiguous())

# Define a GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GNN()

# Forward pass
out = model(graph)
print(out)