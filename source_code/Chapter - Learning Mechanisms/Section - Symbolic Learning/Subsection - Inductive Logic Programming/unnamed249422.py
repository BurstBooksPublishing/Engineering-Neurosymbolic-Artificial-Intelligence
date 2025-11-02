from neuro_symbolic import NeuralSymbolicModel, ILPRuleLayer

# Load pre-trained neural network model
neural_model = NeuralSymbolicModel.load('animal_classifier.h5')

# Define ILP rules learned from data
ilp_rules = [
    "mammal(X) :- has_fur(X), gives_milk(X).",
    "bird(X) :- has_feathers(X), lays_eggs(X)."
]

# Add ILP rule layer to the neural model
rule_layer = ILPRuleLayer(rules=ilp_rules)
neural_model.add_layer(rule_layer)

# Use the enhanced model for prediction
features = {'has_fur': True, 'gives_milk': True}
prediction = neural_model.predict(features)

print(f"Predicted class: {prediction}")