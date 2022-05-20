Daniel Borisov, Jonas Lehnert, and Bianca Granato

We included severfal .py files which incorporated the training and testing of the different models.

#################################################################################
- plotting.py 
plots all of the results shown in the report. Utilizes all the pickle files included in the .zip, which are history files saved from our training experiments of the different models. These files are located in the "PickleFiles" folder in the .zip

#################################################################################
Files required: 
    'train_max_y.csv': labels for training data. CSV must contain a column "Label".
    'train_max_x': pickled file with training images.
    'test_max_x': pickled file with test images.

A short description of the from-scratch CNN files:

- SimpleModel.py
The simple CNN that was also applied to the original CNN. To change from using original MNIST to modified MNIST, please change "mnistFlag = 0" to "mnistFlag = 1"

- Final.py
The Best performing CNN using the SGD Nesterov optimizer

- FinalNolastlayer.py
The CNN using SGD Nesterov with only one layer of 512 nodes. For 1024 nodes, please change line 70 to and switch "512" to "1024".

- lessfilters-adam.py
CNN model using the adam optimizer

- sgdnesterov-l2reg.py
Using regularization in the CNN. Default is L1, for L2 please change l1 to l2 on the reg variable for l2.

- Nodropout.py
Using the CNN model without any dropout.


#################################################################################
     transferLearning.py
Imports VGG16 and applies it to new data set through transfer learning. All required files should be in the same directory as the script.
Computer with GPU recommended. If device doesn't have GPUs, lines 127-129 should be ommited.
Files required: 
    'train_max_y.csv': labels for training data. CSV must contain a column "Label".
    'train_max_x': pickled file with training images.
    'test_max_x': pickled file with test images.
Code was tested with these packages:
numpy: 1.17.3
pandas: 0.25.2
pickle: 4.0
sklearn: 0.21.3
tensorflow: 2.0.0
####################################################################
plotAugmentationDistribution.py
Plots the distribution of data in the training and validation datasets and examples of images pre- and post-augmentation. 
Files required: 
    'train_max_y.csv': labels for training data. CSV must contain a column "Label".
    'train_max_x': pickled file with training images.
    'test_max_x': pickled file with test images.
Code was tested with these packages:
matplotlib: 3.1.1
numpy: 1.17.3
pandas: 0.25.2
sklearn: 0.21.3
