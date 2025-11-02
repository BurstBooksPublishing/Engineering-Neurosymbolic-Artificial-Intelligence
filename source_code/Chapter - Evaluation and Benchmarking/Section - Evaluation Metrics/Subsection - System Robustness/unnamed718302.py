def execute_command(action, object):
    if action == "turn on" and object == "light":
        return "Light turned on"
    elif action == "increase" and object == "volume":
        return "Volume increased"
    elif action == "open" and object == "window":
        return "Window opened"
    else:
        return "Command not recognized"

# Example usage
sample_command = "turn on the light"
predicted_components = clf.predict(vectorizer.transform([sample_command]))[0]
response = execute_command(predicted_components[0], predicted_components[1])

print(response)