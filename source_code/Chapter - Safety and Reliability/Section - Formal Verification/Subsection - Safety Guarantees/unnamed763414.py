import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Example neural network
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Generate adversarial examples during training
def generate_adversarial(data, labels, epsilon=0.1):
    # Perturb the data slightly to simulate an attack
    perturbations = epsilon * np.sign(tf.gradients(model.loss, model.input))
    adversarial_data = data + perturbations
    return adversarial_data, labels

# Custom training loop to include adversarial examples
def train_with_adversarial(model, data, labels, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        adversarial_data, adversarial_labels = generate_adversarial(data, labels)
        model.train_on_batch(adversarial_data, adversarial_labels)
        model.train_on_batch(data, labels)

# Example data
data = np.random.random((100, 10))
labels = np.random.randint(0, 3, 100)

train_with_adversarial(model, data, labels)