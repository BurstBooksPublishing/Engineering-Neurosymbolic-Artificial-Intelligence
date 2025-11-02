import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_classes=2, n_clusters_per_class=1, 
                           weights=[0.99, 0.01], n_features=20, 
                           n_samples=1000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple neural network
nn = MLPClassifier(random_state=42)
nn.fit(X_train, y_train)

# Define a simple symbolic rule: if feature 5 is above 0.5, classify as 1, else 0
def symbolic_rule(x):
    return 1 if x[5] > 0.5 else 0

# Combine neural network with symbolic rule
def neuro_symbolic_model(x):
    nn_output = nn.predict([x])[0]
    symbolic_output = symbolic_rule(x)
    return nn_output if nn_output == symbolic_output else symbolic_output

# Evaluate the model
predictions = [neuro_symbolic_model(x) for x in X_test]
accuracy = np.mean(predictions == y_test)

print(f"Accuracy: {accuracy}")