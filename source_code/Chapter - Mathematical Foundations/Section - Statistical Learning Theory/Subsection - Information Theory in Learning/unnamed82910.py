import numpy as np
from sklearn.feature_selection import mutual_info_classif
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D
from keras.datasets import mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple CNN model
input_layer = Input(shape=(28, 28, 1))
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
flattened = Flatten()(conv1)
output_layer = Dense(10, activation='softmax')(flattened)
model = Model(inputs=input_layer, outputs=output_layer)

# Train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Extract features from the last convolutional layer
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('conv2d').output)
features = feature_extractor.predict(x_train)

# Flatten features for mutual information calculation
flattened_features = features.reshape(features.shape[0], -1)

# Calculate mutual information between features and labels
mi = mutual_info_classif(flattened_features, y_train)
print("Mutual Information: ", mi)