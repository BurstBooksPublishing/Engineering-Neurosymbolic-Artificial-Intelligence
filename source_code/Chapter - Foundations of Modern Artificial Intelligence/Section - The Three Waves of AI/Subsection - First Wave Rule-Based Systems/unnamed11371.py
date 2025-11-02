import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic data
data = np.random.random((1000, 2))
labels = np.array([rule_based_check(amount, countries) for amount, countries in data])

# Split data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

# Build and train the neural network
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10, batch_size=10)

# Evaluate the model
model.evaluate(test_data, test_labels)