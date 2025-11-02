def symbolic_reasoning(objects):
    if 'cat' in objects and 'dog' in objects:
        return 'Conflict'  # Just a simple example of a rule
    elif 'cat' in objects:
        return 'Cat present'
    else:
        return 'No cat'