import torch
import torch.nn as nn
import torch.optim as optim

class LogicalImplication(nn.Module):
    def __init__(self):
        super(LogicalImplication, self).__init__()
        self.linear = nn.Linear(4, 2)  # Assuming 2D representations for A, B, and C

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Represent True as [1, 0] and False as [0, 1]
true = torch.tensor([1.0, 0.0])
false = torch.tensor([0.0, 1.0])

# Inputs for A and B
input_tensor = torch.cat((true, false), 0)  # Concatenate representations of A and B

# Initialize the model and optimizer
model = LogicalImplication()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Assume we know the output for C (True in this case)
target = true

# Training loop
for _ in range(100):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

print("Learned output for 'If True and False, then':", output)