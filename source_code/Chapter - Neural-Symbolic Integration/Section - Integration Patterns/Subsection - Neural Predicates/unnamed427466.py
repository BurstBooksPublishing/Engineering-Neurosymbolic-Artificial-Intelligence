import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    return model

# Instantiate the model
model = create_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assume 'train_images' and 'train_labels' are available from a preprocessed dataset
# where labels are 1 if a cat is present and 0 otherwise
# model.fit(train_images, train_labels, epochs=10, batch_size=64)