import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


import warnings
warnings.filterwarnings("ignore")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# preprocess the imgs
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255
# to normalize img to 0-1 range

# reshape the dimensions 28x28x1
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
# one-hot encoding of target variable

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# train model
model.fit(x_train, y_train, epochs=10, validation_split=0.3)

# Save the model
model.save('m.h5')

# Load the model using Keras method
loaded_model = keras.models.load_model('m.h5')




























# import numpy as np
# import matplotlib.pyplot as plt
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

# import pickle

# import warnings
# warnings.filterwarnings("ignore")

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train.shape, y_train.shape, x_test.shape, y_test.shape

# #preprocess the imgs
# x_train = x_train.astype(np.float32)/255
# x_test = x_test.astype(np.float32)/255
# #to normalize img to 0-1 range

# #reshape the dimensions 28x28x1
# x_train = x_train.reshape((60000, 28, 28, 1))
# x_test = x_test.reshape((10000, 28, 28, 1))

# y_train = keras.utils.to_categorical(y_train)
# y_test = keras.utils.to_categorical(y_test)
# #one-hot encoding of target variable

# model = Sequential()

# model.add(Conv2D(32, (3,3), input_shape = (28,28,1), activation= 'relu'))
# model.add(MaxPool2D((2,2)))

# model.add(Conv2D(64, (3,3), activation= 'relu'))
# model.add(MaxPool2D((2,2)))

# model.add(Flatten())

# model.add((Dropout(0.25)))

# model.add(Dense(10, activation= 'softmax'))

# model.compile(optimizer='adam', loss= keras.losses.categorical_crossentropy, metrics= ['accuracy'])

# #callbacks


# #earlystoppping

# #train model
# model.fit(x_train,y_train, epochs=10, validation_split=0)


# pickle.dump(model, open('model.pkl','wb'))
# modelR = pickle.load(open('model.pkl','rb'))



