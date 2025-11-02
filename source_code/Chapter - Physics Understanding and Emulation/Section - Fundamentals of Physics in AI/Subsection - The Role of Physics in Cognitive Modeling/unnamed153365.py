import tensorflow as tf

# Create a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(None, 15)),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model with an entropy-based loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Generate some synthetic data
import numpy as np

data = np.random.random((1000, 15))
labels = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit(data, labels, epochs=10)