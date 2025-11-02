from pslpython.model import Model
from pslpython.predicate import Predicate
from pslpython.rule import Rule

# Define predicates
Friend = Predicate('Friend', closed=False, size=2)
Likes = Predicate('Likes', closed=False, size=2)

# Create a model
model = Model()

# Add predicates to the model
model.add_predicate(Friend)
model.add_predicate(Likes)

# Define rules with weights
model.add_rule(Rule('0.7: Friend(A, B) & Likes(B, C) -> Likes(A, C) ^2'))

# Load data and infer
model.load_data({
    'Friend': [('Alice', 'Bob'), ('Bob', 'Charlie')],
    'Likes': [('Bob', 'IceCream'), ('Charlie', 'IceCream')]
})

# Perform inference
results = model.infer()
print(results['Likes'])