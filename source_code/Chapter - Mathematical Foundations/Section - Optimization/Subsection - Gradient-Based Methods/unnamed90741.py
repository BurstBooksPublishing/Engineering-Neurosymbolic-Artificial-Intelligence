def differentiable_and(x, y):
    # Approximate AND operation using a differentiable function
    return tf.sigmoid(10 * (x + y - 1.5))

# Use in a model
input_x = Input(shape=(1,))
input_y = Input(shape=(1,))
output = differentiable_and(input_x, input_y)

logical_model = Model(inputs=[input_x, input_y], outputs=output)
logical_model.compile(optimizer='adam', loss='binary_crossentropy')