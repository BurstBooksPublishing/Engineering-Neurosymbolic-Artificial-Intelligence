import networkx as nx

# Create a Directed Acyclic Graph
rule_graph = nx.DiGraph()

# Add nodes representing rules
rule_graph.add_node("rule1", function=lambda x: x > 10)
rule_graph.add_node("rule2", function=lambda x: x % 2 == 0)

# Add edges with dependencies
rule_graph.add_edge("rule1", "rule2")

# Execute rules in topological order
def execute_rules(graph, input_value):
    for node in nx.topological_sort(graph):
        rule_func = graph.nodes[node]['function']
        if not rule_func(input_value):
            return False
    return True

# Example usage
input_value = 12
result = execute_rules(rule_graph, input_value)
print("Rules satisfied:", result)