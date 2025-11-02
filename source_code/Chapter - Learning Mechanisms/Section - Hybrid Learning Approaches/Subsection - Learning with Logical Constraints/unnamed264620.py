import tensorflow as tf

def custom_loss(y_true, y_pred):
    # Standard categorical crossentropy for classification accuracy
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Adding a penalty term for class overlap: assume class 0 and class 1 should not overlap
    penalty = tf.reduce_sum(tf.abs(y_pred[:, 0] * y_pred[:, 1]))
    
    return loss + 0.1 * penalty  # Weighted sum of the original loss and the penalty

# Build and compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])