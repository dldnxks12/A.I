## On google colab environment

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten 
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Import Dataset
train_images = np.load("/content/drive/MyDrive/Colab Notebooks/2022-1 딥러닝/Pet/imageData_25.npy", allow_pickle=True)
train_labels = np.load("/content/drive/MyDrive/Colab Notebooks/2022-1 딥러닝/Pet/imageDataLabel_25.npy", allow_pickle=True)

# Data augmentation 

aug1 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.1) ])

aug28 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.2) ])

aug2 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.3) ])

aug29 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.4) ])

aug3 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.5) ])

aug30 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.6) ])

aug4 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.7) ])

aug31 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.8) ])

aug5 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.9) ])

aug6 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug7 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.8), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug8 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.3) ])

aug9 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.8), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.3) ])

aug10 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.2), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug11 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.8), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug12 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.2), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.3) ])

aug13 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.8), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.3) ])

aug14 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug15 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug16 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.5),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug17 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.7),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug18 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.9),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug19 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.1) ])

aug20 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.1) ])

aug21 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.1) ])

aug22 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.4),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.1) ])

aug23 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.5),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.1) ])

aug24 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.6),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.1) ])

aug25 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.7),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.1) ])

aug26 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.8),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.1) ])

aug27 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.9),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.1) ])

aug32 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug33 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.6), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug34 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.3) ])

aug35 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.3) ])

aug36 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.3), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug37 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.6), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug38 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.3), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.3) ])

aug39 = Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
                                 tf.keras.layers.experimental.preprocessing.RandomRotation(0.6), 
                                 tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.3) ])

aug40 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.1), 
                   tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug41 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.1), 
                   tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.8) ])

aug42 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.1), 
                    tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.5) ])

aug43 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.3), 
                   tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug44 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.3), 
                   tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.8) ])

aug45 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.3), 
                    tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.5) ])

aug46 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.6), 
                   tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug47 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.6), 
                   tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.8) ])

aug48 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.6), 
                    tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.5) ])

aug49 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.9), 
                    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,0.5) ])

aug50 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.9), 
                    tf.keras.layers.experimental.preprocessing.RandomZoom(0.3,0.8) ])

aug51 = Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.9), 
                    tf.keras.layers.experimental.preprocessing.RandomZoom(0.5,0.5) ])


functions_list = [aug1, aug2, aug3, aug4, aug5, aug6, aug7, aug8, aug9, aug10, aug11, aug12, aug13,   \
                  aug14, aug15, aug16, aug17, aug18, aug19, aug20, aug21, aug22, aug23, aug24, aug25, \
                  aug26, aug27, aug28, aug29, aug30, aug31, aug32, aug33, aug34, aug35, aug36, aug37, \
                  aug38, aug39, aug40]

print(" Before Augmentation ")
print(train_images.shape)
print(train_labels.shape)

train_temp = train_images
train_temp_label = train_labels

for function in functions_list:  
  temp = train_temp
  temp_label = train_temp_label
  train_images = np.append(train_images, function(temp), axis = 0 )
  train_labels = np.append(train_labels, temp_label,     axis = 0 )
  
print(" After Augmentation ")  
print(train_images.shape)
print(train_labels.shape)  

# Split dataset train -> train + valid 
train_images, valid_images, train_labels, valid_labels = train_test_split( train_images, train_labels, test_size = 0.1, shuffle = True,  random_state = 105)

# Preprocessing Image  
# Normalize train , valid images for training 
train_size = train_images.shape[0] 
valid_size = valid_images.shape[0] 
train_images =train_images.reshape(train_size, 64, 64, 3)
valid_images =valid_images.reshape(valid_size, 64, 64, 3)

train_images = train_images / 255.0
valid_images = valid_images / 255.0

# Preprocessing Label
from sklearn.preprocessing import OneHotEncoder
enc1 = OneHotEncoder()
enc2 = OneHotEncoder()
train_labels_new = train_labels.reshape(-1, 1)
valid_labels_new = valid_labels.reshape(-1, 1)

enc1.fit(train_labels_new) 
enc2.fit(valid_labels_new) 

train_labels_onehot = np.array(enc1.transform(train_labels_new).toarray()) # train label encoding
valid_labels_onehot = np.array(enc2.transform(valid_labels_new).toarray()) # valid label encoding

print("one-hot encoding train shape is ",train_labels_onehot.shape)  
print("one-hot encoding valid shape is ",valid_labels_onehot.shape)  

# Base Model 
# Define Model
with tf.device('/gpu:0'):
  characters = 25 # 20 개 class 
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)))
  model.add(layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(128, (3, 3), activation='relu')) 
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))  
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(characters, activation='softmax'))

  model.summary()
  
  with tf.device('/gpu:0'):
  model.compile(optimizer='SGD',
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                )
  batch_size = 64 
  history=model.fit(train_images, train_labels_onehot, batch_size=batch_size, epochs = 40, verbose=1)
  
with tf.device('/gpu:0'):
  score = model.evaluate(valid_images, valid_labels_onehot, verbose = 1)
  
#saved_model이라는 이름으로 해당 모델을 저장한다.
model.save("/content/drive/MyDrive/Colab Notebooks/2022-1 딥러닝/Pet/saved_model")

# Load saved model and check ...
loaded_model = keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/2022-1 딥러닝/Pet/saved_model")
with tf.device('/gpu:0'):
  score = loaded_model.evaluate(valid_images, valid_labels_onehot, verbose = 1)
