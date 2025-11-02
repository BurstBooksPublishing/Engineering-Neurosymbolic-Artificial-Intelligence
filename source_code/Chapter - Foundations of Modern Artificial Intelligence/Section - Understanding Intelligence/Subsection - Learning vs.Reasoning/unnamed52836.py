def recommend_activity(image):
    scene = predict_scene(image)
    activities = suggest_activity(scene)
    return f"For a {scene}, consider: {activities}"