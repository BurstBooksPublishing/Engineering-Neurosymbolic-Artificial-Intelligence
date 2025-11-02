import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network with a logic layer
class NeuralSymbolicModel(nn.Module):
    def __init__(self):
        super(NeuralSymbolicModel, self).__init__()
        self.linear = nn.Linear(10, 5)
        self.logic = nn.Linear(5, 2)  # This could represent a simple logical rule

    def forward(self, x):
        x = torch.relu(self.linear(x))
        x = torch.sigmoid(self.logic(x))  # Apply logical reasoning in the network
        return x

# Sample data
X = torch.rand(10, 10)  # Random features
y = torch.randint(0, 2, (10, 2)).float()  # Random binary labels

# Training the model
model = NeuralSymbolicModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

print("Training complete")