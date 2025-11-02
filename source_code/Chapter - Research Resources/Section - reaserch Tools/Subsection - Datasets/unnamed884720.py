from graph_knowledge_base import BiologicalGraph
from symbolic_reasoner import query_graph

# Initialize the graph from the dataset
graph = BiologicalGraph('path_to_dataset')

# Example query: "What does enzyme E interact with?"
query_result = query_graph(graph, 'enzyme E')
print(query_result)