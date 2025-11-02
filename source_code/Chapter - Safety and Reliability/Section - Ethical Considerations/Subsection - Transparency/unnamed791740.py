import numpy as np

# Simulated neural network output
neural_output = np.array([0.8, 0.1, 0.1])  # Probabilities for three diseases

# Symbolic rules encoded as functions
def diagnose_disease_A(prob):
    if prob > 0.75:
        return "Disease A diagnosed based on rule 1"
    return "Rule 1 not applicable"

def diagnose_disease_B(prob):
    if prob > 0.75:
        return "Disease B diagnosed based on rule 2"
    return "Rule 2 not applicable"

def diagnose_disease_C(prob):
    if prob > 0.75:
        return "Disease C diagnosed based on rule 3"
    return "Rule 3 not applicable"

# Applying symbolic reasoning
diagnosis_A = diagnose_disease_A(neural_output[0])
diagnosis_B = diagnose_disease_B(neural_output[1])
diagnosis_C = diagnose_disease_C(neural_output[2])

print(diagnosis_A)  # Output: Disease A diagnosed based on rule 1
print(diagnosis_B)  # Output: Rule 2 not applicable
print(diagnosis_C)  # Output: Rule 3 not applicable