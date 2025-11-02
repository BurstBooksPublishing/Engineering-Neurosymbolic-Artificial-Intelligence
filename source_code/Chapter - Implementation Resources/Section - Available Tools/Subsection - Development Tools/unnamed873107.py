import sympy as sp

# Define symbolic variables
x, y = sp.symbols('x y')

# Define the equation
expression = x + 2*y

# Solve the equation x + 2y = 15 for y
solution = sp.solve(expression - 15, y)

# Print the solution
print(solution)