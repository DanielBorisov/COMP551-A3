# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:16:12 2019

@author: Bianca
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Remove variables that won't be necessary?
remove_vars = 1
# %% Define Functions
def augmentImages(x, imgSize = 224, threshold = 0.9):

    # Convert to values to 0 < x < 1
    x = x/max(x[0,:,:].flatten())
    
    # Threshold values
    x[x < threshold] = 0
    
    # Transform images to fit VGG16 size (224 x 224)
    aug_x = np.zeros((x.shape[0],imgSize,imgSize),dtype='float32')
    
    # Find padding length
    padNo = int((imgSize - x.shape[1])/2)
    padCol = np.zeros((padNo,x.shape[1]), dtype='float32')
    padRow = np.zeros((imgSize,padNo), dtype='float32')
    
    # Stack zeros to each side
    for i in range(x.shape[0]):
        tmp = np.vstack((padCol,x[i,:,:],padCol))
        aug_x[i,:,:] = np.hstack((padRow, tmp, padRow))
    return aug_x

def addChannels(x):
    return np.stack((x,x,x),axis = -1)

# %% Import data
# Load data sets
ytrain = pd.read_csv('train_max_y.csv')
xtrain = pd.read_pickle('train_max_x')

# Get training and validation sets
train_x, valid_x, train_y, valid_y = train_test_split(xtrain, 
                                                      ytrain['Label'].values,
                                                      test_size=0.2,
                                                      shuffle=True,
                                                      stratify=ytrain['Label'].values,
                                                      random_state=2)

if remove_vars:
    del train_test_split, ytrain, xtrain
# %% Naive data augmentation
# Augment images by thresholding, converting to 0-1 and padding
aug_train = augmentImages(train_x)
aug_valid = augmentImages(valid_x)

# Add channels to augmented images
aug_train = addChannels(aug_train)
aug_valid = addChannels(aug_valid)

# Add channels to original images
train_x = addChannels(train_x)
valid_x = addChannels(valid_x)

# %% Prepare transfer learning model
'''
Based on code from 
https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
'''
# Import VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers

# Import pre-trained VGG16 with and without top layer
VGG_base = VGG16(weights = 'imagenet', include_top = False, 
                 input_shape = (224,224,3))

VGG_new_size = VGG16(weights = 'imagenet', include_top = False, 
                 input_shape = tuple(np.append(train_x.shape[1:3],3)))

if remove_vars:
    del VGG16

# Freeze all layers except the last 4 for each model
for layer in VGG_base.layers[:5]:
    layer.trainable = False
    
for layer in VGG_new_size.layers[:5]:
    layer.trainable = False

# Create new models for transfer learning (TL) 
VGG_TL_base = Sequential()
VGG_TL_new_size = Sequential()

if remove_vars:
    del Sequential

# Add VGG to TL model
VGG_TL_base.add(VGG_base)
VGG_TL_new_size.add(VGG_new_size)

# Add layers to match new output
VGG_TL_base.add(layers.Flatten())
VGG_TL_base.add(layers.Dense(1024, activation='relu'))
VGG_TL_base.add(layers.Dropout(0.5))
VGG_TL_base.add(layers.Dense(len(np.unique(train_y)), activation='softmax'))

# Add new layers to match new output
VGG_TL_new_size.add(layers.Flatten())
VGG_TL_new_size.add(layers.Dense(1024, activation='relu'))
VGG_TL_new_size.add(layers.Dropout(0.5))
VGG_TL_new_size.add(layers.Dense(len(np.unique(train_y)), activation='softmax'))

# Check summary of model
VGG_TL_base.summary()
VGG_TL_new_size.summary()

# %% Apply TL
from tensorflow.keras.utils import to_categorical
from tensorflow import config

physical_devices = config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config.experimental.set_memory_growth(physical_devices[0], True)

# Compile model, specify model hyperparameters
VGG_TL_base.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', #Tried: adam, rmsprop
            metrics=['accuracy'])
VGG_TL_new_size.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', 
            metrics=['accuracy'])

# Fit models
baseFit = VGG_TL_base.fit(aug_train,to_categorical(train_y), 
                validation_data=(aug_valid, to_categorical(valid_y)),
                epochs = 50)
newFit = VGG_TL_new_size.fit(train_x,to_categorical(train_y), 
                validation_data=(valid_x, to_categorical(valid_y)),
                epochs = 50)

# %% Predict

augImg = 1 # Augment images?
# Import and augment data
xkaggle = pd.read_pickle('test_max_x')
if augImg:
    xkaggle = augmentImages(xkaggle)
xkaggle = addChannels(xkaggle)

# Predict
if augImg:
    testpredictions = VGG_TL_base.predict(xkaggle)
else:
    testpredictions = VGG_TL_new_size.predict(xkaggle)

# Write Prediction file
testSubmitNum = np.argmax(testpredictions, axis=1)
idx = np.arange(10000)
testSubmitDf = pd.DataFrame({'ID': idx, 'Label': testSubmitNum})
testSubmitDf.to_csv('testSubmit.csv', index=False)

# %% Pickle History

import pickle

f = open('basemodel2.pickle', 'wb')
pickle.dump(baseFit.history, f)
f.close()

f = open('newfitmodel2.pickle', 'wb')
pickle.dump(newFit.history, f)
f.close()