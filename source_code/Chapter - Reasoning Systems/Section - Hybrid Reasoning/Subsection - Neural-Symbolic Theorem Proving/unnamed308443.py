import tensorflow as tf
from tensorflow.keras import layers, models

# Example data: (simplified for demonstration)
# Input: features of a triangle, Output: theorem applicable
# Feature: [side lengths], Output: [Pythagorean theorem applicable?]
data = {
    'inputs': [[3, 4, 5], [5, 12, 13], [8, 15, 17]],
    'outputs': [[1], [1], [1]]  # 1 means Pythagorean theorem is applicable
}

# Create a simple neural network model
model = models.Sequential([
    layers.Dense(10, activation='relu', input_shape=(3,)),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Convert data to TensorFlow tensors
inputs = tf.constant(data['inputs'], dtype=tf.float32)
outputs = tf.constant(data['outputs'], dtype=tf.float32)

# Train the model
model.fit(inputs, outputs, epochs=10)