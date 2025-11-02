import tensorflow as tf
from tensorflow.keras.applications import VGG16
import numpy as np

# Load pre-trained VGG16 model for feature extraction
model = VGG16(weights='imagenet', include_top=False)

def extract_features(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

def symbolic_rule(features):
    # Placeholder for simplicity: Assume features[0] is related to having wings
    # and features[1] is related to having a beak
    if features[0] > 0.5 and features[1] > 0.5:
        return "bird"
    else:
        return "not a bird"

# Example usage
image_path = 'path_to_image.jpg'
features = extract_features(image_path)
classification = symbolic_rule(features)

print(f"The image is classified as: {classification}")