import torch
import torch.nn as nn
from torch_geometric.nn import RGCN

class KnowledgeGraphEmbeddingModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(KnowledgeGraphEmbeddingModel, self).__init__()
        self.rgcn = RGCN(num_entities, num_relations, embedding_dim, num_bases=10)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.rgcn(x, edge_index, edge_type)
        x = torch.mean(x, dim=0)  # Average pooling
        out = self.classifier(x)
        return out

# Assume `data` is loaded and preprocessed, including edge lists and edge types
# model = KnowledgeGraphEmbeddingModel(num_entities, num_relations, embedding_dim)
# output = model(data)