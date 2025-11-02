import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Load a pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

def extract_features(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.vgg16.preprocess_input(image_array)
    features = model.predict(image_array)
    return features