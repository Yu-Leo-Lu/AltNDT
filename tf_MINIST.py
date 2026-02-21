# %%
from framework import tfNDT
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from framework.ndtFunc import statModel

dataName = 'MNIST'
# %% utility functions, plot data images and their labels
def display_images(X, Y):
    # display 3 random images of X with their labels Y
    random_indices = np.random.randint(0, X.shape[0], 3)
    plt.figure(figsize=(10, 10))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"Label: {Y[random_indices[i]]}")
        plt.imshow(X[random_indices[i]], cmap='Greys')
    plt.show()

# %% 
def mainPlot(i):
    plt.figure(i)
    plt.plot(ndt_record.epoch_range, E_tree_test * np.ones((ndt.epochs + 1, 1)), color='k', label='DT')
    plt.plot(ndt_record.epoch_range, ndt_record.tClf_error, marker='o', color='b', label='NDT')
    plt.plot(nn_record.epoch_range, nn_record.tClf_error, marker='o', color='g', label='NN')
    plt.plot(nnD_record.epoch_range, nnD_record.tClf_error, marker='o', color='r', label='NN-D')
    plt.plot(nnH_record.epoch_range, nnH_record.tClf_error, marker='o', color='c', label='NN-H')
    plt.plot(nn1_record.epoch_range, nn1_record.tClf_error, marker='o', color='m', label='NN-1')
    plt.plot(nn3_record.epoch_range, nn3_record.tClf_error, marker='o', color='y', label='NN-3')

    # plt.ylim(0, 10)
    plt.xticks(np.arange(ndt.epochs + 1, step=5))
    plt.xlabel('Epochs ')
    plt.ylabel('Classification Error in %')
    plt.title(dataName+' Testing')
    plt.legend(loc=1)
    plt.show()
    plt.savefig('Improved-NDT/main/figure/'+dataName+'_testing.png', format='png', dpi=100)


def avgPlot(i):
    plt.figure(i)
    plt.plot(ndt_avgR.epoch_range, E_tree_test * np.ones((ndt.epochs + 1, 1)), color='k', label='DT')

    plt.plot(ndt_avgR.epoch_range, ndt_avgR.tClf_error, marker='o', color='b', label='NDT')
    plt.plot(nn_avgR.epoch_range, nn_avgR.tClf_error, marker='o', color='g', label='NN')
    plt.plot(nnD_avgR.epoch_range, nnD_avgR.tClf_error, marker='o', color='r', label='NN-D')
    plt.plot(nnH_avgR.epoch_range, nnH_avgR.tClf_error, marker='o', color='c', label='NN-H')
    plt.plot(nn1_avgR.epoch_range, nn1_avgR.tClf_error, marker='o', color='m', label='NN-1')
    plt.plot(nn3_avgR.epoch_range, nn3_avgR.tClf_error, marker='o', color='y', label='NN-3')

    # plt.ylim(0, 10)
    plt.xticks(np.arange(ndt.epochs + 1, step=5))
    plt.xlabel('Epochs ')
    plt.ylabel('Classification Error in %')
    plt.title(dataName+' Testing')
    plt.legend(loc=1)
    plt.show()
    plt.savefig('Improved-NDT/main/figure/'+dataName+'_testing.png', format='png', dpi=100)


def trainPlot(i):
    plt.figure(i)
    # plt.plot(ndt_record.epoch_range, E_tree_train * np.ones((ndt.epochs + 1, 1)), color='k', label='DT')
    # plt.plot(ndt_record.epoch_range, E_treeP_train * np.ones((ndt.epochs + 1, 1)), color='k', linestyle='--',
    #          label='DT-P')
    plt.plot(ndt_record.epoch_range, ndt_record.Cost, marker='o', color='b', label='NDT')
    # plt.plot(ndtP_record.epoch_range, ndtP_record.Cost, marker='x', color='b', linestyle='--', label='NDT-P')
    plt.plot(nn_record.epoch_range, nn_record.Cost, marker='o', color='g', label='NN')
    plt.plot(nnD_record.epoch_range, nnD_record.Cost, marker='o', color='r', label='NN-D')
    plt.plot(nnH_record.epoch_range, nnH_record.Cost, marker='o', color='c', label='NN-H')
    plt.plot(nn1_record.epoch_range, nn1_record.Cost, marker='o', color='m', label='NN-1')
    plt.plot(nn3_record.epoch_range, nn3_record.Cost, marker='o', color='y', label='NN-3')

    # plt.ylim(0, 10)
    plt.xlabel('Epochs')
    plt.ylabel('L2 cost')
    plt.title(dataName+' Training')
    plt.legend()
    plt.show()

# %%
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

# show 3 images of X_train using plt.imshow
# add a title as the label of the image Y_train corresponding to the 3 images
# add options of random 3 images rather than the first 3 images
display_images(X_train, Y_train)

# %%
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_test = Y_test.reshape(Y_test.shape[0], 1)

# %%
X_train_selected, Y_train_selected = X_train, Y_train  # multi-classification
X_test_selected, Y_test_selected = X_test, Y_test
# %%
d_train = tfNDT.DataProcess(X_train_selected, Y_train_selected)
d_test = tfNDT.DataProcess(X_test_selected, Y_test_selected)
d_train.preProcessData()
d_test.preProcessData()

runs = 10  # observe statistical properties

# %%

ndt = tfNDT.NeuralDecisionTreeClassification()
ndt.d_train, ndt.d_test = d_train, d_test
clf = ndt.treeConfig()
y_pred = clf.predict(ndt.d_test.X)
y_pred = y_pred.reshape((len(y_pred), 1))
E_tree_test = np.mean(np.not_equal(y_pred, ndt.d_test.Y)) * 100
y_pred_ = clf.predict(ndt.d_train.X)
y_pred_ = y_pred_.reshape((len(y_pred_), 1))
E_tree_train = np.mean(np.not_equal(y_pred_, ndt.d_train.Y))

ndt.activation1 = tfNDT.r1
ndt.learning_rate = 0.25
ndt.batch_size = 1000

ndt, ndt_record = ndt.train()
ndt_avgR = ndt_record

# ndt_s, ndt_records = [], []
# for _ in range(runs):
#     clf = ndt.treeConfig()
#     ndt, ndt_record = ndt.train()
#     ndt_s.append(ndt)
#     ndt_records.append(ndt_record)
# ndt_avgR = statModel(ndt_records)
# %%

nn = tfNDT.NeuralDecisionTreeClassification()
nn.d_train, nn.d_test = d_train, d_test
nn.learning_rate = 25
nn.batch_size = 1000

nn_s, nn_records = [], []
for _ in range(runs):
    nn.Wb = ndt.Wb
    nn.netConfig(option='randN')
    nn, nn_record = nn.train()
    nn_s.append(nn)
    nn_records.append(nn_record)
nn_avgR = statModel(nn_records)

# %%

nnD = tfNDT.NeuralDecisionTreeClassification()
nnD.d_train, nnD.d_test = d_train, d_test
nnD.learning_rate = 25
nnD.batch_size = 1000

nnD_s, nnD_records = [], []
for _ in range(runs):
    nnD.Wb = ndt.Wb
    nnD.netConfig(option='double')
    nnD, nnD_record = nnD.train()
    nnD_s.append(nnD)
    nnD_records.append(nnD_record)
nnD_avgR = statModel(nnD_records)

# %%

nnH = tfNDT.NeuralDecisionTreeClassification()
nnH.d_train, nnH.d_test = d_train, d_test
nnH.learning_rate = 25
nnH.batch_size = 1000

nnH_s, nnH_records = [], []
for _ in range(runs):
    nnH.Wb = ndt.Wb
    nnH.netConfig(option='half')
    nnH, nnH_record = nnH.train()
    nnH_s.append(nnH)
    nnH_records.append(nnH_record)
nnH_avgR = statModel(nnH_records)
# %%

num_neurons = 2 * ndt.Wb[0].shape[1] + 1
nn1 = tfNDT.OneLayersNetworkClassification(num_neurons=num_neurons)
nn1.d_train, nn1.d_test = d_train, d_test
nn1.learning_rate = 25
nn1.batch_size = 1000

nn1_s, nn1_records = [], []
for _ in range(runs):
    nn1.netConfig()
    nn1, nn1_record = nn1.train()
    nn1_s.append(nn1)
    nn1_records.append(nn1_record)
nn1_avgR = statModel(nn1_records)
# %%

nn3 = tfNDT.ThreeLayersNetworkClassification(
    num_neurons_list=[int(num_neurons / 3), int(num_neurons / 3), num_neurons - 2 * int(num_neurons / 3)])
nn3.d_train, nn3.d_test = d_train, d_test
nn3.learning_rate = 25
nn3.batch_size = 1000

nn3_s, nn3_records = [], []
for _ in range(runs):
    nn3.netConfig()
    nn3, nn3_record = nn3.train()
    nn3_s.append(nn3)
    nn3_records.append(nn3_record)
nn3_avgR = statModel(nn3_records)

# %%
mainPlot(11)
trainPlot(12)
avgPlot(13)
# %%
