def conservation_loss(predicted_theta, predicted_velocity, actual_energy):
    predicted_energy = pendulum.total_energy(predicted_theta, predicted_velocity)
    return (predicted_energy - actual_energy).pow(2)

# Example usage
pendulum = Pendulum(mass=1.0, length=1.0)

theta = torch.tensor([0.5], requires_grad=True)
velocity = torch.tensor([0.1], requires_grad=True)

actual_energy = pendulum.total_energy(theta, velocity)

# Predict next state (simplified example)
predicted_theta = theta + 0.1 * velocity
predicted_velocity = velocity - 0.1 * pendulum.gravity * torch.sin(theta)

# Calculate loss
loss = conservation_loss(predicted_theta, predicted_velocity, actual_energy)
loss.backward()