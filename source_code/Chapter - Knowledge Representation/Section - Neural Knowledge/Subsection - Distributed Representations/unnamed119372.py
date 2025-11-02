import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential

# Sample vocabulary size and embedding dimensions
vocab_size = 10000
embedding_dim = 16

# Build the model
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Assume `train_data` and `train_labels` are preprocessed and ready to use
# train_data = ... (your dataset here)
# train_labels = ... (labels here)

# Train the model
model.fit(train_data, train_labels, epochs=10)