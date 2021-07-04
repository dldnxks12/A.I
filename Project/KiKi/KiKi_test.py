import numpy as np
import keras
from keras import models
from keras import layers

train_images = np.load("/content/drive/MyDrive/Colab Notebooks/Project/train.npy") # train_image 불러오기 -> 경로 재설정 필수 

reconstruct = keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/Project/saved_model") # 가중치 불러오기 -> 경로 재설정 필수 

train_sample = np.expand_dims(train_images[0], axis = 0) # 1개씩 predict 할 때 차원 맞춰주기

pred = reconstruct.predict(train_sample)

pred # One-Hot Encoding 된 
