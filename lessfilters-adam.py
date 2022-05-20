import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import pickle

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


ytrain = pd.read_csv('train_max_y.csv')

xtrain = pd.read_pickle('train_max_x')

#super_threshold_indices = xtrain < 255
#xtrain[super_threshold_indices] = 0

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


#xtrain = (xtrain - xtrain.min())/(xtrain.max() - xtrain.min())

#ytraincat = tf.keras.utils.to_categorical(ytrain['Label'].values)

#X_train = xtrain[0:40000][:][:].reshape(40000,128,128#,1)
#X_test = xtrain[40000:50000][:][:].reshape(10000,128,128,1)

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
model.add(tf.keras.layers.SpatialDropout2D(0.2))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding = 'same'))#,  kernel_regularizer=reg))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding = 'same'))#, kernel_regularizer=reg))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.SpatialDropout2D(0.2))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding = 'same'))#, kernel_regularizer=reg))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding = 'same'))#, kernel_regularizer=reg))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Fitting Data Generator')

#aug.fit(X_train, augment=True)

print('Fitting model')

#model.fit_generator(aug.flow(X_train, y_train, batch_size=1), validation_data=(X_valid, y_valid), epochs=35, use_multiprocessing=True)
twotwothreeadam = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=300, use_multiprocessing=True)

f = open('twotwothreeadam.pickle', 'wb')
pickle.dump(twotwothreeadam.history, f)
f.close()

xkaggle = pd.read_pickle('test_max_x')


## TEST
#forpredictions = model.predict(X_train)
#testsco = model.evaluate(X_valid, y_valid)
#forvalid = model.predict(X_valid)
#randof = KNeighborsClassifier(n_neighbors = 500, n_jobs=-1)
#randof.fit(forpredictions, y_train)
#randoval = randof.score(forvalid, y_valid)
#print(randoval)

#super_threshold_indices = xkaggle < 255
#xkaggle[super_threshold_indices] = 0
#xkaggle = (xkaggle - xkaggle.min())/(xkaggle.max() - xkaggle.min())

xkaggle = xkaggle.reshape(10000,128,128,1)

testpredictions = model.predict(xkaggle)

# Write Prediction file
test_x = pd.read_pickle('test_max_x')
testSubmit = testpredictions
testSubmitNum = np.argmax(testSubmit, axis=1)
id = np.arange(10000)
testSubmitDf = pd.DataFrame({'ID': id, 'Label': testSubmitNum})
testSubmitDf.to_csv('testSubmit.csv', index=False)
