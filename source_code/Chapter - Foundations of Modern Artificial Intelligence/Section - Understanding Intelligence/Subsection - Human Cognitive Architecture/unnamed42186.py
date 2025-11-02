def symbolic_reasoning(entities):
    rules = {
        'greeting': 'Hello, how are you?',
        'weather': 'Today is a sunny day.',
        'interest': 'I enjoy learning about AI.'
    }

    for entity in entities:
        if entity in rules:
            print(f"Interpreted meaning: {rules[entity]}")

# Example output from neural network (simulated)
nn_output = ['greeting', 'weather', 'interest']
symbolic_reasoning(nn_output)