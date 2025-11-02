import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

# Sample text data
texts = [
    'Hello, how are you?',
    'Today is a sunny day.',
    'I enjoy learning about AI.'
]

# Text vectorization
vectorizer = TextVectorization(max_tokens=1000, output_sequence_length=10)
vectorizer.adapt(texts)

# Build the neural network model
model = Sequential([
    vectorizer,
    Embedding(input_dim=1000, output_dim=64),
    LSTM(64),
    Dense(1)
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()