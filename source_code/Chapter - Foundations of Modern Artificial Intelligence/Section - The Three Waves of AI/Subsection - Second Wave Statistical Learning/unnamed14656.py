import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the neural network
model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)),
    Flatten(),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Predict using the trained model
predictions = model.predict(X_test)