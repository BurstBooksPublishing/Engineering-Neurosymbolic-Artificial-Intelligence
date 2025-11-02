import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and optimizer
model = SimpleCNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Implementing Elastic Weight Consolidation (EWC)
class EWC(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._compute_precision_matrices()

    def _compute_precision_matrices(self):
        precision_matrices = {}
        for n, p in self.params.items():
            p.data.zero_()
            precision_matrices[n] = torch.zeros_like(p)

        # Using the dataset to estimate the Fisher Information matrix
        for input, target in self.dataset:
            self.model.zero_grad()
            output = self.model(input)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()

            for n, p in self.params.items():
                precision_matrices[n].data += p.grad.data  2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n])  2
            loss += _loss.sum()
        return loss

# Training with EWC
def train(model, optimizer, data_loader, ewc, importance=1000):
    model.train()
    for input, target in data_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = nn.functional.nll_loss(output, target) + importance * ewc.penalty(model)
        loss.backward()
        optimizer.step()