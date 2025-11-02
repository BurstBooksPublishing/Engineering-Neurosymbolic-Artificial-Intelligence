# Import necessary libraries
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load pre-trained neural network model
model = load_model('fruit_classifier_model.h5')

# Load an image and preprocess it for the model
img = image.load_img('apple.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict features of the image
features = model.predict(img_array)

# Symbolic rule generation (EBL component)
def generate_rule(features):
    if features[0] > 0.5:  # Assuming the first feature is shape and 0.5 is the threshold for round shapes
        return "If shape > 0.5, the fruit is likely round."
    else:
        return "If shape <= 0.5, the fruit is not round."

# Generate and print the rule
rule = generate_rule(features)
print(rule)