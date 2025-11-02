import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

# Sample data
documents = [
    "Deep learning achieves state-of-the-art results",
    "Symbolic AI excels at logical reasoning",
    "Hybrid approaches can leverage the best of both worlds"
]
labels = ["Neural Networks", "Symbolic AI", "Hybrid AI"]

# Vectorizing text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Encoding labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Training a simple neural network
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=300)
mlp.fit(X, y)

# Applying symbolic reasoning
def symbolic_reasoning(vector, threshold=0.5):
    # Example symbolic rule: If the projection on 'hybrid' dimension is high, classify as Hybrid AI
    hybrid_index = vectorizer.vocabulary_['hybrid']
    if vector[0, hybrid_index] > threshold:
        return "Hybrid AI"
    return label_encoder.inverse_transform(mlp.predict(vector))[0]

# Test on a new document
test_doc = vectorizer.transform(["An article discussing both neural networks and logical reasoning"])
predicted_label = symbolic_reasoning(test_doc)

print(f"Predicted label: {predicted_label}")