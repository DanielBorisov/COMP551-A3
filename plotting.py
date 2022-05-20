import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def prettyAxes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    matplotlib.rcParams.update({'font.size': 14})

def set_color_cycle(self, clist=None):
    if clist is None:
        clist = rcParams['axes.color_cycle']
    self.color_cycle = itertools.cycle(clist)

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# First plot two two three cnn, compating different optimizers
adam = pd.read_pickle('twotwothreeadam.pickle')
sgd = pd.read_pickle('twotwothreesgd.pickle')
sgdnesterov = pd.read_pickle('twotwothreesgdnesterov.pickle')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
ax1.plot(adam['accuracy'], '--', color=cycle[0])
ax1.plot(sgd['accuracy'], '--', color=cycle[1])
ax1.plot(sgdnesterov['accuracy'], '--', color=cycle[2])
p1 = ax1.plot(adam['val_accuracy'], color=cycle[0], label='adam')
p2 = ax1.plot(sgd['val_accuracy'], color=cycle[1], label='SGD')
p3 = ax1.plot(sgdnesterov['val_accuracy'], color=cycle[2], label='SGD Nesterov')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1])

ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epochs')
prettyAxes(ax1)

ax2.plot(adam['accuracy'], '--', color=cycle[0])
ax2.plot(sgd['accuracy'], '--', color=cycle[1])
ax2.plot(sgdnesterov['accuracy'], '--', color=cycle[2])
p1 = ax2.plot(adam['val_accuracy'], color=cycle[0], label='adam')
p2 = ax2.plot(sgd['val_accuracy'], color=cycle[1], label='SGD')
p3 = ax2.plot(sgdnesterov['val_accuracy'], color=cycle[2], label='SGD Nesterov')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1])
ax2.set_xlabel('Epochs')
prettyAxes(ax2)
plt.xlim((250, 300))
plt.ylim((0.85, 1))
plt.tight_layout()
fig.savefig('accuraciesOptimizers.png', format='png', dpi=600)

# Plot performance of VGG net, using augmented data vs non-augmented data
transferLearning = pd.read_pickle('basemodel.pickle')
transferLearningAugmented = pd.read_pickle('newfitmodel.pickle')

fig, (ax1) = plt.subplots(1, 1, figsize=(4.5, 3))
ax1.plot(transferLearning['accuracy'], '--', color=cycle[0])
ax1.plot(transferLearningAugmented['accuracy'], '--', color=cycle[1])
ax1.plot(transferLearning['val_accuracy'], color=cycle[0], label='Basic VGG')
ax1.plot(transferLearningAugmented['val_accuracy'], color=cycle[1], label='Augmented VGG')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1])
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epochs')
prettyAxes(ax1)
plt.tight_layout()
plt.ylim((0.8, 1))
fig.savefig('transferLearning.png', format='png', dpi=600)



# Plot performances of regularized CNNs
l1Regularized = pd.read_pickle('twotwothreesgdnestl1.pickle')
l2Regularized = pd.read_pickle('twotwothreesgdnestl2.pickle')
noDropout = pd.read_pickle('twotwothreesgdnestnodropouts.pickle')
noLastLayer = pd.read_pickle('twotwothreesgdnesterovnoLastLayer.pickle')
noLastLayer1024 = pd.read_pickle('twotwothreesgdnesterovnoLastLayer1024.pickle')

fig, (ax2, ax1, ax3) = plt.subplots(3, 1, figsize=(4.5, 10))
ax1.plot(sgdnesterov['accuracy'], '--', color=cycle[0])
ax1.plot(l1Regularized['accuracy'], '--', color=cycle[1])
ax1.plot(l2Regularized['accuracy'], '--', color=cycle[2])
ax1.plot(sgdnesterov['val_accuracy'], color=cycle[0], label='SGD Nesterov')
ax1.plot(l1Regularized['val_accuracy'], color=cycle[1], label='L1 Regularized')
ax1.plot(l2Regularized['val_accuracy'], color=cycle[2], label='L2 regularized')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1])
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_xlim((0, 100))
prettyAxes(ax1)

ax2.plot(sgdnesterov['accuracy'], '--', color=cycle[0])
ax2.plot(noDropout['accuracy'], '--', color=cycle[1])
ax2.plot(sgdnesterov['val_accuracy'], color=cycle[0], label='SGD Nesterov')
ax2.plot(noDropout['val_accuracy'], color=cycle[1], label='No Dropout')
ax2.legend(['SGD Nesterov', 'No Dropout'])
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1])
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_xlim((0, 100))
prettyAxes(ax2)

ax3.plot(sgdnesterov['accuracy'], '--', color=cycle[0])
ax3.plot(noLastLayer['accuracy'], '--', color=cycle[1])
ax3.plot(noLastLayer1024['accuracy'], '--', color=cycle[3])
ax3.plot(sgdnesterov['val_accuracy'], color=cycle[0], label='SGD Nesterov')
ax3.plot(noLastLayer['val_accuracy'], color=cycle[1], label='No Last Layer (512)')
ax3.plot(noLastLayer1024['val_accuracy'], color=cycle[3], label='No Last Layer (1024)')
ax3.legend(['SGD Nesterov', 'No Last Layer (512)', 'No Last Layer (1024)'])
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles[::-1], labels[::-1])
ax3.set_ylabel('Accuracy')
ax3.set_xlabel('Epochs')
ax3.set_xlim((0, 100))
prettyAxes(ax3)

plt.tight_layout()
fig.savefig('structuralComparison.png', format='png', dpi=600)


# Plot performance of a network that works well on MNIST
originalMnist = pd.read_pickle('originalMnist.pickle')
modifiedMnist = pd.read_pickle('simpleModel.pickle')

fig, (ax1) = plt.subplots(1, 1, figsize=(4.5, 3))
ax1.plot(originalMnist['accuracy'], '--', color=cycle[0])
ax1.plot(modifiedMnist['accuracy'], '--', color=cycle[1])
ax1.plot(originalMnist['val_accuracy'], color=cycle[0], label='Original MNIST')
ax1.plot(modifiedMnist['val_accuracy'], color=cycle[1], label='Modified MNIST')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1])
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epochs')
prettyAxes(ax1)
plt.tight_layout()
plt.ylim((0, 1))
fig.savefig('simpleModel.png', format='png', dpi=600)

