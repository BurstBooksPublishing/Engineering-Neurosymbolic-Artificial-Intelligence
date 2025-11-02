def infer_affordances(object):
    if object == 'chair':
        return ['sitting']
    elif object == 'switch':
        return ['turning on', 'turning off']
    elif object == 'cup':
        return ['holding liquids']
    else:
        return []

# Example usage
object_recognized = 'chair'
affordances = infer_affordances(object_recognized)
print(f"The {object_recognized} affords: {affordances}")