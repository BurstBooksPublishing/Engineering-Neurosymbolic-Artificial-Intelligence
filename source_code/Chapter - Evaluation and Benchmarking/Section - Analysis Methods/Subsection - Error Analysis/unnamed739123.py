class AnimalClassifier(KnowledgeEngine):
    @Rule(Animal(category='bat'))
    def is_mammal(self):
        print("It's a mammal.")

    @Rule(Animal(category='bird'))
    def is_bird(self):
        print("It's a bird.")