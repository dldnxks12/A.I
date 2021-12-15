import tensorflow as tf
import numpy as np

x = np.random.randn(10)
y = x*10 + 10

# tensorflow no function
W = tf.Variable(10)
b = tf.Variable(10)

for i in range(300):
    hypothesis = W*x + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))

    W_gradient = tf.reduce_mean((hypothesis - y)*x)
    b_gradient = tf.reduce_mean(hypothesis - y)

    W.assign_sub(learning_rate*W_gradient)
    b.assign_sub(learning_rate*b_gradient)


# tensorflow with Gradient 계산

for i in range(300):
    with tf.GradientTape as tape:
        hypothesis = W*x + b
        cost = tf.reduce_mean(tf.square(hypothesis - y))

    W_grad, b_grad = tape.gradient(cost , [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

# tensorflow with Gradient 계산 + Gradient Update

for i in range(300):
    with tf.GradientTape as tape:
        hypothesis = W * x + b
        cost = tf.reduce_mean(tf.square(hypothesis - y))

    grads = tape.gradient(cost, [W, b])
    optimizer = tf.optimizers.Adam(learning_rate)
    optimizer.apply_gradient(zip(cost, grads))

# tensorflow with Keras

model = Sequential()
model.add(Dense(1, input_shape = (1, ), activation = 'linear'))
model.summary()
model.compile(optimizer = SGD(learning_rate = 0.001), loss = 'mse')




