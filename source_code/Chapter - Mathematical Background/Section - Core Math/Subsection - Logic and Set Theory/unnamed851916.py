from sympy import symbols, And

# Define symbols
animal = symbols('animal')
has_hair = symbols('has_hair')
produces_milk = symbols('produces_milk')
is_mammal = symbols('is_mammal')

# Define rule
mammal_rule = And(has_hair, produces_milk)

# Assume an animal has hair and produces milk
assumptions = {has_hair: True, produces_milk: True}

# Check if the animal is a mammal
is_mammal = mammal_rule.subs(assumptions)

print(f"Is the animal a mammal? {is_mammal}")