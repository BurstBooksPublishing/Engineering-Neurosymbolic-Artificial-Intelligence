import torch
from torch import nn
from torch.nn import functional as F

# Define a simple neural network for text classification
class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=50)
        self.rnn = nn.LSTM(input_size=50, hidden_size=100, num_layers=1, batch_first=True)
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Instantiate the model
model = TextClassifier()

# Define a symbolic rule as a Python function
def fairness_constraint(outputs):
    # Placeholder for fairness constraint: outputs should be balanced across classes
    class_counts = outputs.sum(0)
    return torch.abs(class_counts[0] - class_counts[1]) < 0.1 * outputs.size(0)

# Generate some synthetic data
input_data = torch.randint(0, 1000, (100, 10))
labels = torch.randint(0, 2, (100,))

# Forward pass
outputs = model(input_data)
probabilities = F.softmax(outputs, dim=1)

# Check property (fairness constraint)
property_verified = fairness_constraint(probabilities)
print("Property Verified:", property_verified)