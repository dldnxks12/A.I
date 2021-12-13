import tensorflow as tf

#######################################  Basic
# Graph 생성하는 과정
# tf.constant : Gradient 계산을 하지 않을 상수
x1 = tf.constant([1,2,3,4]) # 바뀌지 않는 값
x2 = tf.constant([1,2,3,4])
result = tf.multiply(x1, xt)

# 그래프 실행하는 과정
with tf.Session() as sess:  # Session Open
    # method 1
    output = sess.run(result)
    # method 2
    output = result.eval()

    print(output)


########################################### Tensorflow v1

X = tf.placeholder("float") # 값을 받을 바구니 생성
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(), name = "W") # 바뀔 수 있는 값
b = tf.Variable(np.random.randn(), name = "b")

hypothesis = X*W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y)) # MSE

# Gradient 계산
W_gradient = tf.reduce_mean((hypothesis - Y)*X)
b_gradient = tf.reduce_mean(hypotehsis - Y)

learning_rate = 0.01
W_updata= W.assign_sub(learning_rate*W_gradient)
b_updata= b.assign_sub(learning_rate*b_gradient)

sess = tf.Session()
sess.run(tf.global_variable_initializer())

for i in range(300):
    sess.run([W_update, b_update], feed_dict = { X : x, Y : y})
    cost_val, W_val, b_val = sess.run([cost, W, b], feed_dict = {X:x,Y:y})

sess.close()

