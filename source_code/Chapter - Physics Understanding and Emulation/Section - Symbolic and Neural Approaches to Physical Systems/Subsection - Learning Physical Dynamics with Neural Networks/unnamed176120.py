import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class DynamicsGNN(MessagePassing):
    def __init__(self):
        super(DynamicsGNN, self).__init__(aggr='mean')  # "Mean" aggregation.
        self.lin = torch.nn.Linear(3, 2)

    def forward(self, x, edge_index):
        # x holds the current state of each node (e.g., position and velocity)
        # edge_index holds the connectivity of the graph (which nodes interact)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j denotes the features of neighboring nodes, which are used to update nodes' states
        return self.lin(x_j)

# Example graph data (simple chain of masses connected by springs)
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
x = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.float)  # Example node features (position, velocity)

graph_data = Data(x=x, edge_index=edge_index.t().contiguous())
model = DynamicsGNN()

# Forward pass through the GNN
output = model(graph_data.x, graph_data.edge_index)