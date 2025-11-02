import networkx as nx

# Create a new graph
G = nx.DiGraph()

# Add nodes with attributes
G.add_node('blue_ball', attributes={'color': 'blue', 'shape': 'ball'})
G.add_node('red_cube', attributes={'color': 'red', 'shape': 'cube'})
G.add_node('table', attributes={'color': 'brown', 'shape': 'rectangle'})

# Add edges with relationships
G.add_edge('blue_ball', 'table', relationship='on')
G.add_edge('red_cube', 'table', relationship='on')

# Visualize the graph
nx.draw(G, with_labels=True)