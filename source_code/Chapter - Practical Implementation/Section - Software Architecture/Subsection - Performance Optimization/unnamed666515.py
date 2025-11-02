import numpy as np

# Simulated output from a neural network
neural_output = np.array([0.8, 0.1, 0.1])

# Threshold to interpret neural output as symbolic data
threshold = 0.7

# Function to convert neural output to symbolic format
def neural_to_symbolic(output, threshold):
    symbolic_output = ['True' if value > threshold else 'False' for value in output]
    return symbolic_output

# Convert and print symbolic output
symbolic_data = neural_to_symbolic(neural_output, threshold)
print(symbolic_data)