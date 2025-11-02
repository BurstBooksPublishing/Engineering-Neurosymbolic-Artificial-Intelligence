import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define a simple CNN model
def create_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # Assuming 3 classes: square, triangle, circle
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Symbolic rules for generating descriptions
def generate_description(shape):
    descriptions = {
        0: "This is a square.",
        1: "This is a triangle.",
        2: "This is a circle."
    }
    return descriptions.get(shape, "Unknown shape.")

# Train and use the CNN
cnn_model = create_cnn()

# Normally you would train the model here and load real image data
# For this example, assume the model is already trained and we're using dummy data
dummy_image = np.zeros((64, 64, 3))  # Placeholder for an actual image

shape_index = np.argmax(cnn_model.predict(np.array([dummy_image])))
description = generate_description(shape_index)
print(description)