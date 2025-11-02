import numpy as np
import cv2
from keras.models import load_model

# Load a pre-trained CNN model for object recognition
model = load_model('path_to_cnn_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Assuming model expects 224x224 input
    img = img.astype('float32')
    img /= 255.0  # Normalizing
    return np.expand_dims(img, axis=0)

def predict_image_class(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    return np.argmax(predictions)  # Returning the index of the highest probability class

# Example usage
image_path = 'path_to_image.jpg'
predicted_class = predict_image_class(image_path)
print(f"Predicted class index: {predicted_class}")