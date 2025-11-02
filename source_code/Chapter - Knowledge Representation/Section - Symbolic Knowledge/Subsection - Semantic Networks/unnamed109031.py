import networkx as nx
import matplotlib.pyplot as plt

# Create a new semantic network
G = nx.DiGraph()

# Add nodes with attributes
G.add_node("Cat", type="Animal", class="Mammal")
G.add_node("Dog", type="Animal", class="Mammal")
G.add_node("Parrot", type="Animal", class="Bird")

# Add edges with relationships
G.add_edge("Cat", "Dog", relationship="similar")
G.add_edge("Cat", "Parrot", relationship="different")

# Visualize the network
plt.figure(figsize=(6, 4))
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='black', font_weight='bold')
plt.show()