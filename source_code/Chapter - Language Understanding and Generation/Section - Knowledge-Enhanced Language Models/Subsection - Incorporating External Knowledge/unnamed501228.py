import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network for classification
class AnimalClassifier(nn.Module):
    def __init__(self):
        super(AnimalClassifier, self).__init__()
        self.fc = nn.Linear(10, 2)  # Assume input features are of size 10

    def forward(self, x):
        return self.fc(x)

# Sample data (features for 'Dog' and 'Not an animal')
features = torch.tensor([
    [0.5, 0.2, 0.1, 0.1, 0.3, 0.6, 0.1, 0.2, 0.5, 0.1], 
    [0.1, 0.1, 0.2, 0.3, 0.4, 0.1, 0.6, 0.7, 0.1, 0.2]
])
labels = torch.tensor([1, 0])  # 1 for animal, 0 for not an animal

# Instantiate model, loss, and optimizer
model = AnimalClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop with logical rule incorporated
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(outputs, labels)

    # Logical rule: if input is 'Dog', output must be 'Animal'
    logical_rule_loss = (outputs[0, 1] - 1)  2  # Encourage class '1' for 'Dog'
    
    total_loss = loss + logical_rule_loss
    total_loss.backward()
    optimizer.step()