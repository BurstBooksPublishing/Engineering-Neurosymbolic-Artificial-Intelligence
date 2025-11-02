def symbolic_reasoning(objects, query):
    """ A simple symbolic reasoning function that processes objects and a query. """
    if query == "count":
        return len(objects)
    elif query == "list":
        return [obj.type for obj in objects]
    else:
        raise ValueError("Unsupported query type")

# Example usage
objects_detected = [Object(type='Car'), Object(type='Bicycle')]
query_result = symbolic_reasoning(objects_detected, "list")
print(query_result)  # Output: ['Car', 'Bicycle']