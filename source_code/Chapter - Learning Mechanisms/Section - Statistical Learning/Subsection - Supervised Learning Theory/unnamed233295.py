import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Now, let's use the trained model in a symbolic reasoning task
# Suppose we have a rule that says if the predicted digit is even, classify it as 'Even', otherwise 'Odd'
def symbolic_reasoning(predictions):
    # Apply the rule to each prediction
    return ['Even' if np.argmax(pred) % 2 == 0 else 'Odd' for pred in predictions]

# Make predictions
probability_model = tf.keras.Sequential([model, layers.Softmax()])
test_predictions = probability_model.predict(test_images)

# Apply symbolic reasoning
test_reasoning_results = symbolic_reasoning(test_predictions)

# Print some results
print(test_reasoning_results[:10])