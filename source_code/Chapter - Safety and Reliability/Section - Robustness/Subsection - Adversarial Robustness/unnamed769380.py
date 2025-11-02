import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load data
train_data = datasets.MNIST(root='./data', train=True, download=True, 
                            transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Define a simple neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Initialize network and optimizer
model = NeuralNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adversarial training loop
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Generate adversarial example here (simple FGSM method)
        epsilon = 0.1
        data_grad = torch.autograd.grad(outputs=output, inputs=data, 
                                        grad_outputs=torch.ones(output.size()).to(device), 
                                        create_graph=True)[0]
        perturbed_data = data + epsilon * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)

        # Re-classify the perturbed image
        output_perturbed = model(perturbed_data)

        # Calculate loss
        loss = nn.CrossEntropyLoss()(output_perturbed, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, epoch)