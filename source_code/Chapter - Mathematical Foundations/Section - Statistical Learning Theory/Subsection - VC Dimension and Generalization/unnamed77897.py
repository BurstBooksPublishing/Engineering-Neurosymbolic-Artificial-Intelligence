import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple CNN model for feature extraction
model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
])

# Extract features
features_train = model_cnn.predict(x_train)
features_test = model_cnn.predict(x_test)

# Define and train a decision tree classifier on the extracted features
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(features_train, y_train.ravel())

# Predict and evaluate
predictions = tree_classifier.predict(features_test)
print("Accuracy:", accuracy_score(y_test, predictions))