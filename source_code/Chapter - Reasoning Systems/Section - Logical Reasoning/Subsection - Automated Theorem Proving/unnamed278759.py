from sympy import symbols, Eq, solve

# Define the angles of a triangle
angle_a, angle_b, angle_c = symbols('angle_a angle_b angle_c')

# Axiom: Sum of angles in a triangle is 180 degrees
triangle_axiom = Eq(angle_a + angle_b + angle_c, 180)

# Example to solve: If angle_a is 90 degrees and angle_b is 45 degrees, find angle_c
example = triangle_axiom.subs({angle_a: 90, angle_b: 45})
angle_c_solution = solve(example, angle_c)

print(f"The third angle is: {angle_c_solution[0]} degrees")