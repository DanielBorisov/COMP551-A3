# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:08:24 2019

@author: Bianca
"""

#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/Bianca/OneDrive - McGill University/School and Research/_COMP551/Assignments/Assignment3')
    
#%% Set up training sets
# Import data
xtrain = pd.read_pickle('train_max_x')
ytrain = pd.read_csv('train_max_y.csv')

# %% Define Functions
def augmentImages(x, imgSize = 224, threshold = 0.9):

    # Convert to values to 0 < x < 1
    x = x/255
    
    # Threshold values
    x[x < threshold] = 0
    
    # Transform images to fit VGG16 size (224 x 224)
    aug_x = np.zeros((imgSize,imgSize),dtype='float32')
    
    # Find padding length
    padNo = int((imgSize - x.shape[1])/2)
    padCol = np.zeros((padNo,x.shape[1]), dtype='float32')
    padRow = np.zeros((imgSize,padNo), dtype='float32')
    
    # Stack zeros to each side
    aug_x = np.vstack((padCol,x,padCol))
    
    return np.hstack((padRow, aug_x, padRow))

def addChannels(x):
    return np.stack((x,x,x),axis = -1)


#%% 


imgTypes = ['Raw', 'Scaled', '+ Thresholding', 'Scaled Coloured', 
            '+ Thresholding','+ Padding']

idx = len(imgTypes)
imgIdx = np.random.choice(np.arange(xtrain.shape[0]),idx)



for i,j in enumerate(imgIdx):
    # Preprocess images
    # Scale
    scaled_x = xtrain[j,:,:]/255
    # Threshold
    thresh_x = xtrain[j,:,:]/255
    thresh_x[thresh_x < 0.9] = 0
    # Coloured
    colour_x = addChannels(scaled_x)
    # Scaled, Thresholded, Coloured
    multi_x = addChannels(thresh_x)
    # Scaled, Thresholded, Coloured, Padded
    aug_x = augmentImages(xtrain[j,:,:])
    aug_x = addChannels(aug_x)
    
    # Plot raw image
    plt.subplot(idx,idx,(i*idx)+1)
    plt.imshow(xtrain[j,:,:,])
    plt.ylabel(imgIdx[i],fontdict={'fontsize':20})
    plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                    labelbottom=False, right=False, left=False, labelleft=False)

    if i == 0:
        plt.title(imgTypes[0],fontdict={'fontsize':20})
    
    # Plot scaled image
    plt.subplot(idx,idx,(i*idx)+2)
    plt.axis('off')
    plt.imshow(scaled_x)
    if i == 0:
        plt.title(imgTypes[1],fontdict={'fontsize':20})
    
    # Plot thresholded image (i.e. all values below threshold set to zero)
    plt.subplot(idx,idx,(i*idx)+3)
    plt.axis('off')
    plt.imshow(thresh_x)
    if i == 0:
        plt.title(imgTypes[2],fontdict={'fontsize':20})
    
    # Plot image with duplicated channel (i.e. coloured)
    plt.subplot(idx,idx,(i*idx)+4)
    plt.axis('off')
    plt.imshow(colour_x)
    if i == 0:
        plt.title(imgTypes[3],fontdict={'fontsize':20})
    
    # Plot thresholded, coloured image
    plt.subplot(idx,idx,(i*idx)+5)
    plt.axis('off')
    plt.imshow(multi_x)
    if i == 0:
        plt.title(imgTypes[4],fontdict={'fontsize':20})
    
    # Plot padded, thresholded and coloured image
    plt.subplot(idx,idx,(i*idx)+6)
    plt.axis('off')
    plt.imshow(aug_x)
    if i == 0:
        plt.title(imgTypes[5],fontdict={'fontsize':20})


#%% Plot distribution of y
from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(xtrain,
                                                      ytrain['Label'].values,
                                                      test_size=0.2,
                                                      shuffle=True,
                                                      stratify=ytrain['Label'].values)
#%%
ax = plt.subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.hist([ytrain['Label'],train_y,valid_y],
         bins=np.arange(max(valid_y+2)),
         density=True,
         align='left',
         label=['Overall','Training Set','Validation Set'])
plt.title('Data distribution', fontdict={'fontsize':20})
plt.xticks(np.arange(max(valid_y+1)))
plt.legend()
