import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.utils import to_categorical

# Sample data: images of fruits and their labels (e.g., 'apple', 'banana')
# For simplicity, assume images are preprocessed and loaded as numpy arrays
images = np.load('fruit_images.npy')
labels = np.load('fruit_labels.npy')

# Encode labels to integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Build a simple CNN to classify fruits based on images
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(len(np.unique(labels)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(images, categorical_labels, epochs=10)