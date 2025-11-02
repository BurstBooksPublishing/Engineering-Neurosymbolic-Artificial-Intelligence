import networkx as nx

# Create a simple knowledge graph
G = nx.Graph()
G.add_edge('Cat', 'Mammal')
G.add_edge('Dog', 'Mammal')
G.add_edge('Mammal', 'Animal')

# Function to check if an entity is a type of animal
def is_animal(entity):
    try:
        path = nx.shortest_path(G, source=entity, target='Animal')
        return True
    except nx.NetworkXNoPath:
        return False

# Example usage
print(is_animal('Dog'))  # Outputs: True
print(is_animal('Car'))  # Outputs: False