import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load a pre-trained model
model = load_model('animal_classifier.h5')

# Load and prepare image
img = image.load_img('lion.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict the animal
predictions = model.predict(img_array)
animal_type = np.argmax(predictions)

# Define symbolic rules
def can_live_in_desert(animal_type):
    # Assuming '1' is the index for 'lion'
    if animal_type == 1:
        return True
    else:
        return False

# Check if the animal can live in the desert
desert_living = can_live_in_desert(animal_type)
print("Can this animal live in the desert?", desert_living)