# Define symbolic rules
def fruit_size(fruit_type):
    size_rules = {
        'apple': 'small',
        'banana': 'large',
        'cherry': 'small'
    }
    return size_rules.get(fruit_type, 'unknown')

# Predict fruit type from an image
def predict_fruit(image):
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_class

# Example usage
test_image = images[0]  # Assume this is an image of a banana
predicted_fruit = predict_fruit(test_image)
fruit_size(predicted_fruit)