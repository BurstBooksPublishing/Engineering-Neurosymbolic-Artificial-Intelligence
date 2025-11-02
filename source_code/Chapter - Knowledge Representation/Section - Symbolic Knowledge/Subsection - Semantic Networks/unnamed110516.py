# Function to retrieve class based on entity
def get_class(entity):
    return G.nodes[entity]["class"]

# Example usage
print("The class of a Cat is:", get_class("Cat"))