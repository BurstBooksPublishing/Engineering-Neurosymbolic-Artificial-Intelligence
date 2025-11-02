import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# Define a simple neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.linear = nn.Linear(2, 1)  # Takes initial velocity and angle

    def forward(self, x):
        return self.linear(x)

# Physics-based symbolic function
def projectile_motion_equations(initial_velocity, angle, time):
    g = 9.81  # acceleration due to gravity
    angle_rad = np.radians(angle)
    x = initial_velocity * np.cos(angle_rad) * time
    y = initial_velocity * np.sin(angle_rad) * time - 0.5 * g * time  2
    return x, y

# Combine neural and symbolic models
def neuro_symbolic_model(data, model, time_points):
    predictions = []
    for velocity, angle in data:
        # Neural network predicts time of flight
        time_of_flight = model(torch.tensor([velocity, angle]).float()).detach().numpy()
        # Symbolic physics calculates trajectory
        trajectory = [projectile_motion_equations(velocity, angle, t) for t in np.linspace(0, time_of_flight, num=time_points)]
        predictions.append(trajectory)
    return predictions

# Example usage
model = NeuralNet()
data = [(50, 45), (30, 60)]  # Example data: list of tuples (velocity, angle)
predicted_trajectories = neuro_symbolic_model(data, model, time_points=10)

for trajectory in predicted_trajectories:
    print(trajectory)