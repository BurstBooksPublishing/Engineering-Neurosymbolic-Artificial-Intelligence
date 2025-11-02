import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Simulated patient data: features might include symptoms encoded numerically
# For simplicity, let's assume 0 = no symptom, 1 = mild, 2 = severe
data = np.array([
    [2, 1, 0],  # Patient 1
    [0, 1, 2],  # Patient 2
    [1, 0, 2]   # Patient 3
])

# Labels: 0 = Healthy, 1 = Flu, 2 = Cold
labels = np.array([1, 2, 1])

# Build a simple neural network model
model = Sequential()
model.add(Dense(5, input_dim=3, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(data, labels, epochs=10, verbose=0)

# Assume the model predicts the following:
predictions = np.array([1, 2, 1])

# Symbolic reasoning
def apply_medical_rules(data, predictions):
    rules_output = predictions.copy()
    for i, (symptoms, predicted) in enumerate(zip(data, predictions)):
        # Rule: if severe cough (index 2 == 2) and no severe fever (index 0 < 2), it's likely a Cold
        if symptoms[2] == 2 and symptoms[0] < 2:
            rules_output[i] = 2  # Cold
        # Additional rules can be implemented here
    return rules_output

# Apply rules
final_diagnoses = apply_medical_rules(data, predictions)
print("Final Diagnoses:", final_diagnoses)