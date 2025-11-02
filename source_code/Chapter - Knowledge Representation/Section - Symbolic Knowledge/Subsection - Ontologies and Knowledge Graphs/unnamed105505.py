import networkx as nx
from pyvis.network import Network

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges from the ontology
G.add_node("my_car", label="Car", color="blue")
G.add_node("my_engine", label="Engine", color="red")
G.add_node("front_left_wheel", label="Wheel", color="green")
G.add_node("front_right_wheel", label="Wheel", color="green")

G.add_edge("my_car", "my_engine", label="hasPart")
G.add_edge("my_car", "front_left_wheel", label="hasPart")
G.add_edge("my_car", "front_right_wheel", label="hasPart")

# Visualize the graph
nt = Network("500px", "500px")
nt.from_nx(G)
nt.show("vehicles.html")