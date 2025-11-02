import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# Create a simple CNN model
def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    return model

model = create_model()
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])