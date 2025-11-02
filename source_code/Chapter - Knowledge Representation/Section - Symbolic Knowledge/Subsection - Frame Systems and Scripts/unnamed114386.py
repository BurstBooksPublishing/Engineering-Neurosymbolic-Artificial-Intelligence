class Frame:
    def __init__(self, name, slots=None):
        self.name = name
        self.slots = slots if slots else {}

    def add_slot(self, slot, value):
        self.slots[slot] = value

    def get_slot(self, slot):
        return self.slots.get(slot, None)

# Neural network output (simulated)
detected_objects = ['sofa', 'table', 'lamp', 'person', 'dog']

# Frame instantiation
room_frame = Frame("Room")
room_frame.add_slot("Furniture", [obj for obj in detected_objects if obj in ['sofa', 'table', 'lamp']])
room_frame.add_slot("People", [obj for obj in detected_objects if obj == 'person'])
room_frame.add_slot("Pets", [obj for obj in detected_objects if obj == 'dog'])

print(room_frame.slots)