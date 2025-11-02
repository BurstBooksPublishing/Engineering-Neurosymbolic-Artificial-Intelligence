import numpy as np

# Simulated output from a neural network
object_probabilities = {'cat': 0.9, 'dog': 0.1, 'tree': 0.05}

# Symbolic rules encoded as functions
def likely_cat_scene(objects):
    if objects['cat'] > 0.8:
        return True
    return False

def likely_dog_scene(objects):
    if objects['dog'] > 0.8:
        return True
    return False

# Reasoning based on neural output
scene = ''
if likely_cat_scene(object_probabilities):
    scene = 'This is probably a cat scene.'
elif likely_dog_scene(object_probabilities):
    scene = 'This is probably a dog scene.'
else:
    scene = 'Uncertain scene composition.'

print(scene)