from sklearn.tree import DecisionTreeClassifier, export_text
from keras.models import Sequential
from keras.layers import Dense

# Train a simple neural network
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(x_train.reshape(-1, 784), y_train, epochs=5)

# Use the neural network's softmax outputs as input for the decision tree
nn_predictions = nn_model.predict(x_train.reshape(-1, 784))

# Train decision tree on the outputs
tree_model = DecisionTreeClassifier(max_depth=5)
tree_model.fit(nn_predictions, y_train)

# Print the decision tree rules
tree_rules = export_text(tree_model, feature_names=[f'class_{i}' for i in range(10)])
print(tree_rules)