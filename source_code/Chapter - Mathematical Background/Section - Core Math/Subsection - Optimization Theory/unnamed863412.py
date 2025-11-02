def apply_symbolic_rule(neural_output, threshold):
    if neural_output > threshold:
        # Apply symbolic rule
        return neural_output * 0.9  # Example modification by the rule
    else:
        return neural_output

# Example usage
neural_output = 0.85
threshold = 0.8
adjusted_output = apply_symbolic_rule(neural_output, threshold)

print(f"Adjusted Output: {adjusted_output}")