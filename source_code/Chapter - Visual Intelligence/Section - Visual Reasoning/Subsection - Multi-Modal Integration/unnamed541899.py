import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load ResNet50 model pre-trained on ImageNet data
model = ResNet50(weights='imagenet')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

def simple_symbolic_reasoner(predictions, question):
    # A very basic example of symbolic reasoning
    if 'dog' in question:
        for pred in predictions:
            if 'dog' in pred[1]:
                return f"Yes, there is a {pred[1]} in the image."
        return "No, there are no dogs in the image."

# Example usage
image_path = 'path_to_image.jpg'
question = 'Is there a dog in the image?'
predictions = predict_image(image_path)
answer = simple_symbolic_reasoner(predictions, question)
print(answer)