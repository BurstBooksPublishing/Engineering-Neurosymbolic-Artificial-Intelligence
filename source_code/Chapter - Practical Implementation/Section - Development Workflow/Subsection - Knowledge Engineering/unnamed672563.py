from pyswip import Prolog

# Initialize Prolog engine
prolog = Prolog()

# Define knowledge base
prolog.assertz("mammal(bear)")
prolog.assertz("mammal(tiger)")
prolog.assertz("has_fur(bear)")
prolog.assertz("has_fur(tiger)")
prolog.assertz("has_claws(bear)")
prolog.assertz("has_claws(tiger)")
prolog.assertz("carnivore(X) :- mammal(X), has_claws(X), has_fur(X)")

# Query knowledge base
list(prolog.query("carnivore(X)"))