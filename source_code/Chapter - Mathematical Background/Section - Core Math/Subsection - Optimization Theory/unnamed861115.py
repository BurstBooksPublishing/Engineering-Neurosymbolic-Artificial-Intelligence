import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Initialize the model, loss function, and optimizer
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy data
inputs = torch.randn(10)
true_outputs = torch.randn(1)

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, true_outputs)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")