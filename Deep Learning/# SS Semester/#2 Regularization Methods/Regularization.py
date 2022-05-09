import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten
from keras import backend as K
from matplotlib import pyplot

import tensorflow as tf

batch_size = 28
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(X_train, Y_train), (x_test,y_test) = mnist.load_data()

x_validation = X_train[1000:1200]
y_validation = Y_train[1000:1200]
large_x_train = X_train[3000:5000]
large_y_train = Y_train[3000:5000]
x_train = X_train[:1000]
y_train = Y_train[:1000]

print("x_validation shape:", x_validation.shape)
print("y_validation shape:", y_validation.shape)

# Channel order modification
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    large_x_train = large_x_train.reshape(large_x_train.shape[0], 1, img_rows, img_cols)
    x_validation = x_validation.reshape(x_validation.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    large_x_train = large_x_train.reshape(large_x_train.shape[0], img_rows, img_cols, 1)
    x_validation = x_validation.reshape(x_validation.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Data type modification
x_train = x_train.astype('float32')
large_x_train = large_x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_test = x_test.astype('float32')

# Data value modification
x_train /= 255
large_x_train /= 255
x_validation /= 255
x_test /= 255

# convert class vectors to binary class matrices  ------- y_label data -> one hot vector
from keras import utils as np_utils # keras == 2.40 and tensorflow == 2.3.0 version ...
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
large_y_train = keras.utils.np_utils.to_categorical(large_y_train, num_classes)
y_validation = keras.utils.np_utils.to_categorical(y_validation, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

assert large_x_train.shape[0]==2000
assert large_y_train.shape[0]==2000
assert x_train.shape[0]==1000
assert y_train.shape[0]==1000
assert x_validation.shape[0]==200
assert y_validation.shape[0]==200

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("large_x_train shape:", large_x_train.shape)
print("large_y_train shape:", large_y_train.shape)
print("x_validation shape:", x_validation.shape)
print("y_validation shape:", y_validation.shape)

model = Sequential([keras.layers.Flatten(input_shape = (28, 28)),   # 28 x 28 image
                    keras.layers.Dense(1024, activation = 'relu'),
                    keras.layers.Dense(1024, activation = 'relu'),
                    keras.layers.Dense(1024, activation = 'relu'),
                    keras.layers.Flatten(input_shape = (1024, 1)),  # 1024 x 1
                    keras.layers.Dense(10, activation = 'softmax'), # 1024 -> 10
                    ])
model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
weights = model.get_weights()

print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)

history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#initialize weights
model.set_weights(weights)

# Data pumping
large_model_history=model.fit(large_x_train, large_y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation))

score = model.evaluate(x_test, y_test, verbose=0)
print('Largemodel Test loss:', score[0])
print('Largemodel Test accuracy:', score[1])

# Dropout
dropout_model = Sequential([keras.layers.Flatten(input_shape = (28, 28)),
                            keras.layers.Dense(1024, activation = 'relu'),
                            keras.layers.Dense(1024, activation = 'relu'),
                            keras.layers.Dense(1024, activation = 'relu'),
                            keras.layers.Dropout(0.5),
                            keras.layers.Flatten(input_shape = (1024, 1)),
                            keras.layers.Dense(10, activation = 'softmax'),
                            ])
dropout_model.summary()

dropout_model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

dropout_model_history=dropout_model.fit(large_x_train, large_y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation))

score = dropout_model.evaluate(x_test, y_test, verbose=0)
print('Dropout model Test loss:', score[0])
print('Dropout model Test accuracy:', score[1])


# Batch Normalization

bn_model = Sequential()

bn_model.add(keras.layers.Flatten(input_shape = (28,28)))
bn_model.add(keras.layers.Dense(1024 ))
bn_model.add(keras.layers.BatchNormalization())
bn_model.add(keras.layers.Activation('relu'))
bn_model.add(keras.layers.Dense(1024))
bn_model.add(keras.layers.BatchNormalization())
bn_model.add(keras.layers.Activation('relu'))
bn_model.add(keras.layers.Dense(1024))
bn_model.add(keras.layers.BatchNormalization())
bn_model.add(keras.layers.Activation('relu'))
bn_model.add(keras.layers.Flatten(input_shape = (1024,1)))
bn_model.add(keras.layers.Dense(10, activation = 'softmax'))
bn_model.summary()

bn_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
bn_model_history=bn_model.fit(large_x_train, large_y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation))

score = bn_model.evaluate(x_test, y_test, verbose=0)
print('BN model Test loss:', score[0])
print('BN model Test accuracy:', score[1])

final_model = Sequential()

# Mixture
final_model.add(keras.layers.Flatten( input_shape = (28,28)))
final_model.add(keras.layers.Dense(1024))
final_model.add(keras.layers.BatchNormalization())
final_model.add(keras.layers.Activation('relu'))
final_model.add(keras.layers.Dense(1024))
final_model.add(keras.layers.BatchNormalization())
final_model.add(keras.layers.Activation('relu'))
final_model.add(keras.layers.Dense(1024))
final_model.add(keras.layers.BatchNormalization())
final_model.add(keras.layers.Activation('relu'))
final_model.add(keras.layers.Dropout(0.5))
final_model.add(keras.layers.Flatten(input_shape = (1024,1)))
final_model.add(keras.layers.Dense(10, activation = 'softmax'))

final_model.summary()


final_model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
final_model_history=final_model.fit(large_x_train, large_y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation))

score = final_model.evaluate(x_test, y_test, verbose=0)
print('Final model Test loss:', score[0])
print('Final model Test accuracy:', score[1])