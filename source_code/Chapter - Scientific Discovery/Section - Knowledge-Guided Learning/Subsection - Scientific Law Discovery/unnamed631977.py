from sympy import symbols, Eq, solve
import numpy as np

l, T = symbols('l T')

# Assume the neural network predicted T = 2 * pi * sqrt(l / g)
# where g is approximately 9.81, but we pretend we don't know this value
g = symbols('g')
equation = Eq(T, 2 * np.pi * (l / g)0.5)

solved_g = solve(equation.subs({T: periods.mean(), l: lengths.mean()}), g)
print(f"Estimated g: {solved_g[0]}")