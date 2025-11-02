import tensorflow as tf

def pde_loss(model, inputs, outputs):
    # Calculate the gradient of outputs with respect to inputs
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)

    # Derivative of predictions with respect to inputs
    dp_dx = tape.gradient(predictions, inputs)

    # Loss is the mean squared error of the PDE residual
    residual = dp_dx + outputs - tf.exp(-inputs)
    return tf.reduce_mean(tf.square(residual))