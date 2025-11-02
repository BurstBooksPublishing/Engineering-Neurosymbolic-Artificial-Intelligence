from tensorflow.keras.layers import SimpleRNN

# Adding a SimpleRNN layer to the model
model.add(SimpleRNN(64, return_sequences=True, return_state=True))
model.summary()