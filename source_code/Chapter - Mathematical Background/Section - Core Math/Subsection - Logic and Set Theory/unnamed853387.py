import torch
import torch.nn.functional as F

# Define embeddings
animal_embedding = torch.randn(1, 10)  # Random embedding vector for an animal
feature_embeddings = {
    'has_hair': torch.randn(1, 10),
    'produces_milk': torch.randn(1, 10)
}
category_embeddings = {
    'mammal': torch.randn(1, 10)
}

# Neural network predicts features of the animal
predicted_features = torch.sigmoid(animal_embedding @ torch.stack(list(feature_embeddings.values())).T)

# Apply logical rule: Mammal if has hair and produces milk
mammal_score = F.relu(predicted_features[0, 0] + predicted_features[0, 1] - 1)  # Simplified AND logic

print(f"Mammal score: {mammal_score.item()}")