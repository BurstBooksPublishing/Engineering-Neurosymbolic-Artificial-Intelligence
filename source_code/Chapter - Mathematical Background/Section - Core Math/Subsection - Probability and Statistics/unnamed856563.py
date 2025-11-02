import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Example data: features are [temperature, humidity], label is [suitable (1) or not suitable (0)]
data = np.array([
    [22, 70, 1],
    [28, 60, 1],
    [30, 85, 0],
    [18, 90, 0],
    [25, 65, 1]
])

X = data[:, :2]
y = data[:, 2]

# Neural network for predicting suitability
model = Sequential([
    Dense(2, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=0)

# Predict probabilities
probabilities = model.predict(np.array([[26, 75]]))

print(f"Predicted probability of suitability: {probabilities[0][0]}")