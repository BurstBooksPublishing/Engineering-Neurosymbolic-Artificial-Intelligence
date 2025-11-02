import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Define a simple neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')

# Dummy data
import numpy as np
data = np.random.random((1000, 10))
labels = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit(data, labels, epochs=10, batch_size=32)