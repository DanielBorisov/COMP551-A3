import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import pickle

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

ytrain = pd.read_csv('train_max_y.csv')

xtrain = pd.read_pickle('train_max_x')

train_x, valid_x, train_y, valid_y = train_test_split(xtrain,
                                                      ytrain['Label'].values,
                                                      test_size=0.2,
                                                      shuffle=True,
                                                      stratify=ytrain['Label'].values)

X_train = train_x.reshape(40000,128,128,1)
X_valid = valid_x.reshape(10000,128,128,1)

aug = tf.keras.preprocessing.image.ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.1,
		width_shift_range=0.2,
		height_shift_range=0.2,
		fill_mode="nearest",
        data_format="channels_last",
        zca_whitening=True)

y_train = tf.keras.utils.to_categorical(train_y)
y_valid = tf.keras.utils.to_categorical(valid_y)


model = tf.keras.models.Sequential()
#reg = tf.keras.regularizers.l2(0.0001)
#add layers
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding = 'same', input_shape=(128,128,1)))#kernel_regularizer=reg, input_shape=(128,128,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding = 'same'))#,  kernel_regularizer=reg) )
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding = 'same'))#,  kernel_regularizer=reg) )
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding = 'same'))#,  kernel_regularizer=reg))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding = 'same'))#, kernel_regularizer=reg))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding = 'same'))#, kernel_regularizer=reg))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding = 'same'))#, kernel_regularizer=reg))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

print('Fitting model')

#model.fit_generator(aug.flow(X_train, y_train, batch_size=1), validation_data=(X_valid, y_valid), epochs=35, use_multiprocessing=True)
twotwothreeSGD = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=300, use_multiprocessing=True)

print('Saving history')

f = open('twotwothreesgdnesterov.pickle', 'wb')
pickle.dump(twotwothreeSGD.history, f)
f.close()

print('Predicting test set')

xkaggle = pd.read_pickle('test_max_x')

xkaggle = xkaggle.reshape(10000,128,128,1)

testpredictions = model.predict(xkaggle)

# Write Prediction file
test_x = pd.read_pickle('test_max_x')
testSubmit = testpredictions
testSubmitNum = np.argmax(testSubmit, axis=1)
id = np.arange(10000)
testSubmitDf = pd.DataFrame({'ID': id, 'Label': testSubmitNum})
testSubmitDf.to_csv('testSubmit.csv', index=False)
