import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Generate synthetic data: length (l), angle (theta), and period (T)
data_size = 1000
np.random.seed(42)
lengths = np.random.uniform(0.1, 2.0, data_size)
angles = np.random.uniform(0, np.pi/4, data_size)
periods = 2 * np.pi * np.sqrt(lengths / 9.81)  # Simplified pendulum period formula

# Neural network for regression
model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(np.column_stack((lengths, angles)), periods, epochs=50, verbose=0)