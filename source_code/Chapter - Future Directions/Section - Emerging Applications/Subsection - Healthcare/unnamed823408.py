import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Neural network for predicting diabetes risk
def create_neural_model():
    model = Sequential()
    model.add(Dense(12, input_dim=3, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Symbolic reasoning to interpret neural network output
def symbolic_reasoning(prediction, age, weight):
    risk_factor = prediction[0]
    if age > 45 and weight > 80:
        risk_factor += 0.1  # Increase risk factor based on symbolic rule
    return risk_factor

# Example data (age, weight, blood sugar level)
data = np.array([[50, 85, 180]])
labels = np.array([1])  # 1 indicates diabetes

# Train neural model
neural_model = create_neural_model()
neural_model.fit(data, labels, epochs=10, verbose=0)

# Predict using the neural model
neural_output = neural_model.predict(np.array([[55, 90, 190]]))

# Apply symbolic reasoning
final_prediction = symbolic_reasoning(neural_output, 55, 90)

print(f"Adjusted Risk Factor: {final_prediction}")