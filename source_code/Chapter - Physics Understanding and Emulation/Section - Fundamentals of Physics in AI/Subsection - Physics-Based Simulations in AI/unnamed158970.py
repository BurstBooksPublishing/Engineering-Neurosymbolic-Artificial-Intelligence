def enforce_conservation(predicted_values, conservation_law):
    corrected_values = predicted_values - (sum(predicted_values) - conservation_law)
    return corrected_values