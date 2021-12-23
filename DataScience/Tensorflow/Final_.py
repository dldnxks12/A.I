'''
# Linear Regression

Tensorflow V1
Tensorflow V2
Tensorflow + Keras

'''

# Tensorflow V1

import tensorflow as tf
import numpy as np

x = np.random.randn(10)
y = x*10 + 10

X = tf.placeholder("float32")
Y = tf.placeholder("float32")

W = tf.Variable(10)
b = tf.Variable(10)

hypothesis = W*X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
learning_rate = 0.001

W_gradient = tf.reduce_mean((hypothesis - Y)*X)
b_gradient = tf.reduce_mean(hypothesis - Y)

W_update = W.assign_sub(learning_rate*W_gradient)
b_update = b.assign_sub(learning_rate*b_gradient)

sess = tf.Session()

sess.run(tf.gloabal_variable_initializer)

for i in range(300):
    sess.run([W_update, b_update], feed_dict = {X:x, Y:y})
    cost_val, W_val, b_val = sess.run([cost, W, b], feed_dict = {X:x, Y:y})


# Tensorflow V2 - no function

W = tf.Variable(10)
b = tf.Variable(10)

for i in range(300):
    hypothesis = W*x + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))

    W_grdient = tf.reduce_mean((hypothesis - y)*x)
    b_gradient = tf.reduce_mean(hypothesis - y)

    W.assign_sub(learning_rate*W_gradient)
    b.assign_sub(learning_rate*b_gradient)

# Tensorflow V2 with gradient 계산 functions

W = tf.Variable(10)
b = tf.Variable(10)

for i in range(300):
    with tf.GradientTape as tape:
        hypothesis = W*x + b
        cost = tf.reduce_mean(tf.square(hypothesis - y))

    W_grad, b_grad = tape.gradient(cost, [W,b])
    W.assign_sub(learning_rate*W_grad)
    b.assign_sub(learning_rate(b_grad))

# Tensorflow V2 with gradient 계산 + gradient Update functions

W = tf.Variable(10)
b = tf.Variable(10)

for i in range(300):
    with tf.GradientTape as tape:
        hypothesis = W * x + b
        cost = tf.reduce_mean(tf.square(hypothesis - y))

    grads = tape.gradient(cost, [W, b])
    optimizer = tf.optimizers.Adam(learning_rate)
    optimizer.apply_graidnet(zip(grads, [W, b]))


# Tensorflow V2 with Keras









