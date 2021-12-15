import tensorflow as tf
import random
import numpy as np

x = np.random.randn(n)
y = x*10 + 10

W = tf.Variable(-10)
b = tf.Variable(10)

learning_rate = 0.01

# No Tensorflow function
for i in range(300):
    hypothesis = W*X + b
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    W_grad = tf.reduce_mean((hypothesis - Y) * X)
    b_grad = tf.reduce_mean(hypothesis - Y)

    W_update = W.assign_sub(learning_rate*W_grad)
    b_update = b.assign_sub(learning_rate * b_grad)


# Tensorflow function --- Gradient 계산
# with tf.GradientTape() as tape: // tape.gradient
for i in range(300):
    with tf.GradientTape() as tape: # Gradient 계산 모두 기록
        hypothesis = W*X + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    W_grad, b_grad = tape.gradient(cost, [W,b])

    W.assign_sub(learning_rage*W_grad)
    b.assign_sub(learning_rate*b_grad)

# Tensorflow function --- optimizer

for i in range(300):
    with tf.GradientTape() as tape:
        hypothesis = W*X + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    grads = tape.gradient(cost, [W, b]) # dcost/dW , dcost/db

    optimizer = tf.optimizer.Adam(learning_rate)
    optimizer.apply_gradient(zip(grads, [W, b]))

    


