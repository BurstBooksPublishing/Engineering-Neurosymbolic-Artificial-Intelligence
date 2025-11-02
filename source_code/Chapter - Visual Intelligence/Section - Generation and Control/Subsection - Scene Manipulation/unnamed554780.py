def modify_scene(original_scene, changes):
    # This function applies changes to the scene
    # Assume changes is a dictionary with new positions for objects
    modified_scene = original_scene.copy()
    for obj, pos in changes.items():
        modified_scene[obj] = pos
    return modified_scene

# Example usage
current_scene = {
    'chair': 'position1',
    'table': 'position2'
}

new_positions = {
    'chair': 'position2',
    'table': 'position1'
}

new_scene = modify_scene(current_scene, new_positions)
print(new_scene)