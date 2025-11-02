from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

# Sample data
commands = ["turn on the light", "increase the volume", "open the window"]
components = [["turn on", "light"], ["increase", "volume"], ["open", "window"]]

# Vectorize the commands
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(commands)

# Train a neural network to predict components
clf = MLPClassifier(hidden_layer_sizes=(15,), max_iter=300)
clf.fit(X_train, components)