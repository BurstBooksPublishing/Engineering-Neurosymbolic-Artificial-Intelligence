# Add new attribute to the Cat node
G.add_node("Cat", pet="Yes")

# Function to check if an entity is a pet
def is_pet(entity):
    return G.nodes[entity].get("pet", "No")

# Example usage
print("Is a Cat a pet?", is_pet("Cat"))