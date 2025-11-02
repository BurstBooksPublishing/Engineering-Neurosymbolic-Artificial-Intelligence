import numpy as np
from simpleai import logic

# Simulated neural network output
weather_conditions = 'rainy'

# Define symbolic rules
rules = logic.PropKB()
rules.tell(logic.expr('rainy => indoor'))
rules.tell(logic.expr('sunny => outdoor'))

# Reasoning function
def plan_activity(weather):
    if rules.ask(logic.expr(weather)):
        return rules.ask(logic.expr(weather))
    else:
        return 'No suitable activity found.'

# Use the system
activity = plan_activity(weather_conditions)
print(f"Suggested activity for {weather_conditions} weather: {activity}")