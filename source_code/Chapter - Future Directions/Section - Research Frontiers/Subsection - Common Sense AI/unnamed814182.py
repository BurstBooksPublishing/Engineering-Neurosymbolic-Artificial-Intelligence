import sympy as sp

# Define symbols
x, y = sp.symbols('x y')

# Define a symbolic expression that represents a common sense rule
rule = sp.Eq(x, y)  # Rule stating that x should be equal to y

# Example usage of the rule
example = rule.subs({x: 3, y: 3})
print("Is the rule satisfied?", example)