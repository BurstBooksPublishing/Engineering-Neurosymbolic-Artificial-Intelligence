import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

# Sample data: simple arithmetic problems
texts = ["five plus three", "seven minus two", "three times four"]
labels = ["8", "5", "12"]  # The symbolic representations of the answers

# Text vectorization
vectorizer = TextVectorization(output_mode='int')
vectorizer.adapt(texts)

# Model building
model = Sequential([
    vectorizer,
    Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=64),
    LSTM(64),
    Dense(3, activation='softmax')  # Assuming a limited set of outputs for simplicity
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Dummy labels for demonstration
import numpy as np
labels = np.array([0, 1, 2])

# Train the model
model.fit(texts, labels, epochs=10)