import tensorflow as tf

# Define inputs as placeholders with a value between 0 and 1
A = tf.placeholder(tf.float32)
B = tf.placeholder(tf.float32)
C = tf.placeholder(tf.float32)

# Define the logical AND as a sigmoid approximation
def soft_and(x, y):
    return tf.sigmoid((x + y - 1) * 10)

# Define the logical OR as a sigmoid approximation
def soft_or(x, y):
    return tf.sigmoid((x + y) * 10)

# Compute (A AND B) OR C
result = soft_or(soft_and(A, B), C)

# Evaluate the expression
with tf.Session() as sess:
    output = sess.run(result, feed_dict={A: 0.9, B: 0.8, C: 0.1})
    print("Output:", output)