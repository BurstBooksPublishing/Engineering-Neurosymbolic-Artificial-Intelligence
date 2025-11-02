import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load a pre-trained CNN model
model = load_model('path_to_my_cnn_model.h5')

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array_expanded)
    return np.argmax(prediction), np.max(prediction)

def symbolic_representation(object_id, confidence):
    # Mapping from model output IDs to human-readable objects
    objects = {0: 'cat', 1: 'dog', 2: 'bird'}
    object_name = objects.get(object_id, 'Unknown')
    
    if confidence > 0.75:
        return f"Detected(high_confidence, {object_name})"
    else:
        return f"Detected(low_confidence, {object_name})"

# Example usage
object_id, confidence = classify_image('path_to_image.jpg')
symbol = symbolic_representation(object_id, confidence)
print(symbol)