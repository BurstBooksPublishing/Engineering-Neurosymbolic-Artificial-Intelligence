from pyproblog import Problog, Term

# Let's assume 'neural_network_prediction' is a function that returns the probabilities
# of high_workload and personal_issues given an email text
def neural_network_prediction(email_text):
    # Dummy probabilities based on hypothetical neural network analysis
    return 0.7, 0.6

# Email text to analyze
email_text = "Bob has been missing deadlines and seems overwhelmed."
high_workload_prob, personal_issues_prob = neural_network_prediction(email_text)

# Define the ProbLog model
model = f"""
{high_workload_prob}::high_workload(bob).
{personal_issues_prob}::personal_issues(bob).
stressed(X) :- high_workload(X), personal_issues(X).
"""

# Create a Problog engine
engine = Problog()

# Load the model
engine.load(model)

# Query the model
query = Term('stressed', 'bob')
result = engine.query(query)

# Print the probability of Bob being stressed
print(f"Probability of Bob being stressed: {result[query]}")