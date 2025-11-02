# Define symbolic rules
animal_facts = {
    'dog': {'can_fly': False, 'has_fur': True},
    'cat': {'can_fly': False, 'has_fur': True},
    'parrot': {'can_fly': True, 'has_fur': False},
}

# Function to answer questions based on classification
def answer_question(animal, question):
    if question == 'can it fly?':
        return animal_facts[animal]['can_fly']
    elif question == 'does it have fur?':
        return animal_facts[animal]['has_fur']
    else:
        return "Unknown question"

# Example usage
animal_class = classify_image('dog.jpg')  
# Assuming 'dog.jpg' is an image of a dog
print(answer_question('dog', 'can it fly?'))  
# Output: False