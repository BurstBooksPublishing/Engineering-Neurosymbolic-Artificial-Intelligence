from sympy import symbols, And, Or, Implies, Exists, ForAll, Function

# Define predicates
Temperature, Light = symbols('Temperature Light', cls=Function)
sensor, value, device, status = symbols('sensor value device status')

# Define expressions
expr1 = ForAll(sensor, Implies(Temperature(sensor, value) > 75, Light(device, 'on')))
expr2 = Exists(sensor, Temperature(sensor, value) > 75)

# Evaluate expressions in some hypothetical context
context = {
    Temperature('sensor1', 77): True,
    Light('device1', 'on'): True
}

is_light_on = expr1.subs(context)
is_high_temp = expr2.subs(context)

print("Is the light on when temperature is above 75?", is_light_on)
print("Is there a sensor with temperature above 75?", is_high_temp)