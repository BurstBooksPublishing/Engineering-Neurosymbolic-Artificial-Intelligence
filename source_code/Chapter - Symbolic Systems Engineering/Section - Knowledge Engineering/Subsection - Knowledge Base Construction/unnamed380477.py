import torch
import torch.nn as nn
import torch.optim as optim

# Define entities and relation as embeddings
embedding_dim = 10
entity_embeddings = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim)
relation_embedding = nn.Parameter(torch.randn(embedding_dim))

# Assign indices to entities
cat_idx = 0
animal_idx = 1

# Define a simple model to learn the 'is a' relation
class SimpleKBModel(nn.Module):
    def __init__(self):
        super(SimpleKBModel, self).__init__()
        self.entity_embeddings = entity_embeddings
        self.relation_embedding = relation_embedding

    def forward(self, entity_idx):
        entity_vec = self.entity_embeddings(torch.tensor([entity_idx]))
        return entity_vec + self.relation_embedding

model = SimpleKBModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example target: Cat is an Animal
target_vector = entity_embeddings(torch.tensor([animal_idx]))
input_vector = model(cat_idx)

loss = criterion(input_vector, target_vector)
loss.backward()
optimizer.step()