# -*- coding: utf-8 -*-
"""DigitRecognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1glaMbGRpY4Q8tejD1nKiRfz1L7aQWyws
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape, y_train.shape, x_test.shape, y_test.shape

np.unique(y_train, return_counts=True)

x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255
#to normalize img to 0-1 range

#reshaping
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

x_train.shape, x_test.shape

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
#one-hot encoding

#defining layers in my model
model = Sequential([
  Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(28,28,1)),
  Conv2D(32, (5,5), padding='same', activation='relu'),
  MaxPool2D(),
  Dropout(0.25),
  Conv2D(64, (3,3), padding='same', activation='relu'),
  Conv2D(64, (3,3), padding='same', activation='relu'),
  MaxPool2D(strides=(2,2)),
  Dropout(0.25),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(10, activation='softmax')
])

model.summary()

#compile
model.compile(optimizer= keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

x_train.shape

y_train.shape

history = model.fit(x_train, y_train, batch_size = 256, epochs=5, validation_split=0.1)

# model_save = keras.models.load_model("/Users/savanim/Desktop/HAII/")
model_save = model.save('my_model_3DEC.h5')

loaded_model = keras.models.load_model('my_model_3DEC.h5')

"""/Users/savanim/Library/Python/3.9/lib/python/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
"""

score = loaded_model.evaluate(x_test, y_test)


print(f" The model Accuracy is {score[1]*100}")

from keras.preprocessing import image

from PIL import Image
import numpy as np
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

# read image using OpenCV
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28),interpolation = cv2.INTER_AREA)
inverted_img = 255 - img
# display image
plt.imshow(inverted_img, cmap='gray')
plt.title('Input Image')
plt.show()

# preprocess the image for your model
img_array = np.expand_dims(inverted_img, axis=0)
img_array = img_array.astype('float32') / 255.0

img_array.shape
img_array = img_array.reshape(1,28,28,1)

x_test[1].shape

img_array.shape

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
print (prediction)

print(f"The predicted number is: {predicted_class}")
