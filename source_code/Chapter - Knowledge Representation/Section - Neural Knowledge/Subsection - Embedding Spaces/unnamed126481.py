import torch
import torch.nn as nn

# Define entities and relations
entities = ['Cat', 'Dog', 'Mouse']
relations = ['is_a', 'chases']

entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}

embedding_dim = 10  # Define embedding dimension

# Create embedding layers for entities and relations
entity_embedding = nn.Embedding(num_embeddings=len(entities), embedding_dim=embedding_dim)
relation_embedding = nn.Embedding(num_embeddings=len(relations), embedding_dim=embedding_dim)

# Example: Get embeddings for the fact 'Cat chases Mouse'
cat_idx = torch.tensor([entity_to_idx['Cat']])
mouse_idx = torch.tensor([entity_to_idx['Mouse']])
chases_idx = torch.tensor([relation_to_idx['chases']])

cat_emb = entity_embedding(cat_idx)
mouse_emb = entity_embedding(mouse_idx)
chases_emb = relation_embedding(chases_idx)

# A simple neural network to predict the validity of the fact
class FactPredictor(nn.Module):
    def __init__(self):
        super(FactPredictor, self).__init__()
        self.fc = nn.Linear(3 * embedding_dim, 1)

    def forward(self, entity1, relation, entity2):
        # Concatenate the embeddings and pass through a linear layer
        x = torch.cat((entity1, relation, entity2), dim=1)
        x = self.fc(x)
        return torch.sigmoid(x)

predictor = FactPredictor()
fact_validity = predictor(cat_emb, chases_emb, mouse_emb)

print("Predicted validity of 'Cat chases Mouse':", fact_validity.item())