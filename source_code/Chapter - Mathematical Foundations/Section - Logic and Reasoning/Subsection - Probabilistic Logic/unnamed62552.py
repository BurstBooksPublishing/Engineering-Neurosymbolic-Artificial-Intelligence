from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, State

# Define the distributions for each node in the network
smoking = DiscreteDistribution({'smoker': 0.2, 'non-smoker': 0.8})
age = DiscreteDistribution({'young': 0.5, 'old': 0.5})

lung_cancer = ConditionalProbabilityTable(
    [
        ['smoker', 'old', 'yes', 0.1],
        ['smoker', 'young', 'yes', 0.03],
        ['non-smoker', 'old', 'yes', 0.01],
        ['non-smoker', 'young', 'yes', 0.005],
        ['smoker', 'old', 'no', 0.9],
        ['smoker', 'young', 'no', 0.97],
        ['non-smoker', 'old', 'no', 0.99],
        ['non-smoker', 'young', 'no', 0.995]
    ],
    [smoking, age]
)

# Create states for each node
s1 = State(smoking, name="smoking")
s2 = State(age, name="age")
s3 = State(lung_cancer, name="lung_cancer")

# Build the Bayesian network
network = BayesianNetwork("Health Assessment")
network.add_states(s1, s2, s3)
network.add_edge(s1, s3)
network.add_edge(s2, s3)
network.bake()

# Use the network for inference
beliefs = network.predict_proba({'age': 'old'})

print("Probabilities given age is old:")
for state, belief in zip(network.states, beliefs):
    print(state.name, belief)