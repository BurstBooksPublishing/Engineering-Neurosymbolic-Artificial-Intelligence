import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feed-forward neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x

# Initialize the network and optimizer
net = NeuralNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)