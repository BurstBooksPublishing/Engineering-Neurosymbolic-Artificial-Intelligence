import tensorflow as tf
import numpy as np
import sympy as sp

# Define symbolic variables
mass, gravity, height = sp.symbols('m g h')

# Define the symbolic equation: E = m * g * h (potential energy)
energy_equation = mass * gravity * height

# Create a dataset
# Let's assume some arbitrary values for gravity and mass
g_value = 9.81  # Earth gravity in m/s^2
m_value = 2.0   # mass in kg
heights = np.linspace(0, 10, 100)  # heights from 0 to 10 meters
energies = m_value * g_value * heights  # calculate energies

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(heights, energies, epochs=10, verbose=0)

# Counterfactual reasoning: What if gravity is doubled?
new_g_value = 2 * g_value
counterfactual_energies = model.predict(heights)

# Adjust the predicted energies according to the new gravity
adjusted_energies = counterfactual_energies * (new_g_value / g_value)

# Print the original and adjusted energies for comparison
print("Original Energies:", energies[:5])
print("Adjusted Energies:", adjusted_energies[:5])