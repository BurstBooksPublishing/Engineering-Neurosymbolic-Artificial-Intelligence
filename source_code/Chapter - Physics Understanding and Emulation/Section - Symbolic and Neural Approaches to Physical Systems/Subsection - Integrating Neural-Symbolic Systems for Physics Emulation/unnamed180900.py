import torch
import torch.nn as nn

# Define a simple neural network for physics emulation
class PhysicsEmulator(nn.Module):
    def __init__(self):
        super(PhysicsEmulator, self).__init__()
        self.dense1 = nn.Linear(2, 50)  # Assume input is initial velocity and angle
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(50, 1)  # Output is the displacement

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x

# Symbolic rule for the motion equation
def motion_equation(u, t, a=-9.81):
    return u * t + 0.5 * a * t  2

# Custom loss function that incorporates the symbolic rule
def custom_loss(outputs, labels, initial_velocity, time):
    symbolic_loss = torch.mean((outputs - motion_equation(initial_velocity, time))  2)
    return symbolic_loss + nn.MSELoss()(outputs, labels)

# Example usage
model = PhysicsEmulator()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

initial_velocity = torch.tensor([10.0])  # Example initial velocity
time = torch.tensor([2.0])  # Example time
labels = torch.tensor([20.0])  # Example true displacement

# Training loop
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    inputs = torch.tensor([initial_velocity, time]).float()
    outputs = model(inputs)
    loss = custom_loss(outputs, labels, initial_velocity, time)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')