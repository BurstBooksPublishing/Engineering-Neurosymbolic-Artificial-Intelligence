# Assume model.predict(image) returns a list of detected objects and their coordinates
detected_objects = model.predict(some_image)  # Mock function call

# Symbolic logic to determine if one object is left of another
def is_left_of(obj1, obj2):
    x1, y1 = obj1['coords']
    x2, y2 = obj2['coords']
    return x1 < x2

# Applying common sense reasoning
obj1, obj2 = detected_objects[0], detected_objects[1]
common_sense_result = is_left_of(obj1, obj2)

print("Object 1 is to the left of Object 2:", common_sense_result)