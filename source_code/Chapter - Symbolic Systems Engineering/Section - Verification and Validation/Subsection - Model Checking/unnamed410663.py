# Import necessary libraries
import numpy as np
from sklearn.neural_network import MLPClassifier

# Example data: features (text transformed to vectors) and labels
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array(['neutral', 'positive', 'negative', 'neutral'])

# Train a simple neural network classifier
clf = MLPClassifier(random_state=1, max_iter=300).fit(X, y)

# Define a simple symbolic rule
def symbolic_decision(classification):
    if classification == 'negative':
        return 'reject'
    else:
        return 'accept'

# Classify a new piece of text (transformed to a vector)
new_text = np.array([1, 0])
predicted_class = clf.predict(new_text.reshape(1, -1))

# Apply the symbolic rule
decision = symbolic_decision(predicted_class[0])

print(f"Decision: {decision}")