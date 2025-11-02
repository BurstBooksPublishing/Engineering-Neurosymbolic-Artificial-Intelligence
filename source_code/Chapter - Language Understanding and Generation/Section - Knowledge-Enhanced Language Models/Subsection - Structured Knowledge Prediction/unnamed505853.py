import torch
import torch.nn as nn
import dgl
from dgl.data import CitationGraphDataset

# Load a graph dataset
data = CitationGraphDataset('cora')
g = data[0]

# Define a Graph Convolutional Network (GCN)
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
        self.conv2 = dgl.nn.GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

# Initialize the model and optimizer
model = GCN(g.ndata['feat'].shape[1], 16, data.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):
    logits = model(g, g.ndata['feat'])
    loss = nn.CrossEntropyLoss()(logits[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))