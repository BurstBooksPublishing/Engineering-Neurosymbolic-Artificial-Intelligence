import tensorflow as tf
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
data = pd.DataFrame(np.random.randn(100, 3), columns=['X', 'Y', 'Z'])
data['Y'] += 3 * data['X'] + np.random.normal(size=100)
data['Z'] += 2 * data['Y'] + np.random.normal(size=100)

# Use TensorFlow to learn representations
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss='mse')
model.fit(data_scaled, data_scaled, epochs=100, verbose=0)

# Extract learned features
features = model.predict(data_scaled)

# Use pgmpy to discover causal structure
hc = HillClimbSearch(pd.DataFrame(features, columns=['X', 'Y', 'Z']))
best_model = hc.estimate(score=BicScore(pd.DataFrame(features, columns=['X', 'Y', 'Z'])))

print("Edges in the best model:", best_model.edges())