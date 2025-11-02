import torch
import torch.nn as nn

# Define a simple pendulum system
class Pendulum(nn.Module):
    def __init__(self, mass, length, gravity=9.81):
        super(Pendulum, self).__init__()
        self.mass = mass
        self.length = length
        self.gravity = gravity

    def potential_energy(self, theta):
        # h = l - l*cos(theta)
        height = self.length * (1 - torch.cos(theta))
        return self.mass * self.gravity * height

    def kinetic_energy(self, velocity):
        return 0.5 * self.mass * (self.length * velocity)  2

    def total_energy(self, theta, velocity):
        return self.potential_energy(theta) + self.kinetic_energy(velocity)