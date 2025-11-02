from owlready2 import *

onto = get_ontology("http://example.org/onto/vehicles")

with onto:
    class Vehicle(Thing):
        pass

    class hasPart(ObjectProperty):
        domain = {Vehicle}
        range = {Thing}

    class Engine(Thing):
        pass

    class Wheel(Thing):
        pass

    class Car(Vehicle):
        pass

    class Motorcycle(Vehicle):
        pass

# Creating specific instances
my_car = Car("my_car")
my_car.hasPart = {
    Engine("my_engine"),
    Wheel("front_left_wheel"),
    Wheel("front_right_wheel")
}

# Save the ontology to a file
onto.save(file="vehicles.owl", format="rdfxml")