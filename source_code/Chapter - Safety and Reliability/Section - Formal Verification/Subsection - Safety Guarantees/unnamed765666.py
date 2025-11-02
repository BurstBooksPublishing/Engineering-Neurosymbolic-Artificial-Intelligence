from z3 import *

# Define symbolic variables
symptom1 = Bool('symptom1')
symptom2 = Bool('symptom2')
diagnosis = Bool('diagnosis')

# Define rules
rule1 = Implies(symptom1, diagnosis)
rule2 = Implies(symptom2, Not(diagnosis))

# Check for consistency
s = Solver()
s.add(rule1, rule2)

if s.check() == sat:
    print("The system is consistent")
else:
    print("The system has contradictions")