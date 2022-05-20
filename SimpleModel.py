import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, naive_bayes, metrics, ensemble, linear_model
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle

# Flag to test original Mnist data on a very simple CNN
mnistFlag = 1

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_x = pd.read_pickle('train_max_x')
train_y = pd.read_csv('/home/jonas/Documents/McGill/Classes/PhD2/COMP551/Miniproject3/train_max_y.csv')
test_x = pd.read_pickle('test_max_x')


if mnistFlag == 0:
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_x,
                                                                          train_y['Label'],
                                                                          test_size=0.2,
                                                                          shuffle=True,
                                                                          stratify=train_y['Label'],
                                                                          random_state=2)

    train_x = train_x.reshape(40000, 128, 128, 1)
    valid_x = valid_x.reshape(10000, 128, 128, 1)
    input_shape = (128, 128, 1)

    num_classes = 10
    train_y = keras.utils.to_categorical(train_y, num_classes)
    valid_y = keras.utils.to_categorical(valid_y, num_classes)
elif mnistFlag == 1:
    (train_x, train_y), (valid_x, valid_y) = mnist.load_data()
    train_x = train_x.reshape(60000, 28, 28, 1)
    valid_x = valid_x.reshape(10000, 28, 28, 1)
    input_shape = (28, 28, 1)

    num_classes = 10
    train_y = keras.utils.to_categorical(train_y, num_classes)
    valid_y = keras.utils.to_categorical(valid_y, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.BatchNormalization())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.SGD(),
              metrics=['accuracy'])

x = model.fit(train_x, train_y,
          epochs=25,
          validation_data=(valid_x, valid_y))

score = model.evaluate(valid_x, valid_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

if mnistFlag==1:
    f = open('originalMnist.pickle', 'wb')
    pickle.dump(x.history, f)
    f.close()
else:
    f = open('simpleModel.pickle', 'wb')
    pickle.dump(x.history, f)
    f.close()
