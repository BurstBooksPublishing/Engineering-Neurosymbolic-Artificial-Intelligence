import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
data_size = 1000
lengths = np.random.uniform(low=0.5, high=2.5, size=data_size)
gravities = np.random.uniform(low=9.5, high=10.5, size=data_size)
periods = 2 * np.pi * np.sqrt(lengths / gravities)

data = pd.DataFrame({
    'length': lengths,
    'gravity': gravities,
    'period': periods
})

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(
    x=train_data[['length', 'gravity']],
    y=train_data['period'],
    epochs=100,
    verbose=0
)

# Evaluate model
mse = model.evaluate(
    x=test_data[['length', 'gravity']],
    y=test_data['period']
)
print(f"Mean Squared Error: {mse}")