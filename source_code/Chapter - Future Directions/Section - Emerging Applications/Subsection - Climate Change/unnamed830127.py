from sympy import symbols, And

# Define symbols
temperature, humidity = symbols('temperature humidity')
threshold_temp = 30  # example threshold for temperature
threshold_hum = 80   # example threshold for humidity

# Define logic for heatwave warning
heatwave_warning = And(temperature > threshold_temp, humidity > threshold_hum)

# Example check
current_conditions = {temperature: 32, humidity: 85}

if heatwave_warning.subs(current_conditions):
    print("Heatwave Warning Issued")