# %%
from framework import tfNDT
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from framework.ndtFunc import statModel

# %%
dataName = 'CIFAR10'
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
# show 3 images of X_train using plt.imshow
# print Y_train corresponding to the 3 images
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_train[i])
    print(Y_train[i])
plt.show()

X_train = X_train.reshape(50000, 32 * 32 * 3)  # 50000 training samples, each sample is 32x32x3 RGB image
X_test = X_test.reshape(10000, 32 * 32 * 3)  # 10000 training samples

# %%

# labels = [0, 5]  # binary classification between two selected label
# X_train_selected, Y_train_selected = tfNDT.ndtFunc.select_labels(X_train, Y_train, labels)
# X_test_selected, Y_test_selected = tfNDT.ndtFunc.select_labels(X_test, Y_test, labels)
# plt.imshow(X_test_selected[1656, :].reshape([28, 28]), cmap='Greys')
# label = 0  # binary classification of selected label vs others
# X_train_selected, Y_train_selected = X_train, tfNDT.ndtFunc.each_label(Y_train, label)
# X_test_selected, Y_test_selected = X_test, tfNDT.ndtFunc.each_label(Y_test, label)
# %%
X_train_selected, Y_train_selected = X_train, Y_train  # multi-classification
X_test_selected, Y_test_selected = X_test, Y_test
# %%
d_train = tfNDT.DataProcess(X_train_selected, Y_train_selected)
d_test = tfNDT.DataProcess(X_test_selected, Y_test_selected)
d_train.preProcessData()
d_test.preProcessData()

runs = 10

# %%
def mainPlot(i):
    plt.figure(i)
    plt.plot(ndt_record.epoch_range, E_tree_test * np.ones((ndt.epochs + 1, 1)), color='k', label='DT')
    plt.plot(ndt_record.epoch_range, E_treeP_test * np.ones((ndt.epochs + 1, 1)), color='k', linestyle='--',
             label='DT-P')
    plt.plot(ndt_record.epoch_range, ndt_record.tClf_error, marker='o', color='b', label='NDT')
    plt.plot(ndtP_record.epoch_range, ndtP_record.tClf_error, marker='x', color='b', linestyle='--', label='NDT-P')
    plt.plot(nn_record.epoch_range, nn_record.tClf_error, marker='o', color='g', label='NN')
    plt.plot(nnD_record.epoch_range, nnD_record.tClf_error, marker='o', color='r', label='NN-D')
    plt.plot(nnH_record.epoch_range, nnH_record.tClf_error, marker='o', color='c', label='NN-H')
    plt.plot(nn1_record.epoch_range, nn1_record.tClf_error, marker='o', color='m', label='NN-1')
    plt.plot(nn3_record.epoch_range, nn3_record.tClf_error, marker='o', color='y', label='NN-3')

    # plt.ylim(0, 10)
    plt.xlabel('Epochs ')
    plt.ylabel('Classification Error in %')
    plt.title(dataName+' Testing')
    plt.legend()
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
    plt.plot(ndtP_record.epoch_range, ndtP_record.Cost, marker='x', color='b', linestyle='--', label='NDT-P')
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
ndt.learning_rate = 0.5
ndt.batch_size = 1000

ndt, ndt_record = ndt.train()
ndt_avgR = ndt_record
# %%

# ndtP = tfNDT.NeuralDecisionTreeClassification()
# ndtP.d_train, ndtP.d_test = d_train, d_test
# clfP = ndtP.treeConfig(init_frac=0.5)
# yP_pred = clfP.predict(ndtP.d_test.X)
# yP_pred = yP_pred.reshape((len(yP_pred), 1))
# E_treeP_test = np.mean(np.not_equal(yP_pred, ndtP.d_test.Y)) * 100
# yP_pred_ = clfP.predict(ndt.d_train.X)
# yP_pred_ = yP_pred_.reshape((len(yP_pred_), 1))
# E_treeP_train = np.mean(np.not_equal(yP_pred_, ndtP.d_train.Y))
#
# ndtP.activation1 = tfNDT.r1
# ndtP.learning_rate = 50
# ndtP, ndtP_record = ndtP.train()

# %%

nn = tfNDT.NeuralDecisionTreeClassification()
nn.d_train, nn.d_test = d_train, d_test
nn.learning_rate = 50
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
nnD.learning_rate = 50
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
nnH.learning_rate = 50
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
nn1.learning_rate = 50
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
nn3.learning_rate = 50
nn3.batch_size = 1000

nn3_s, nn3_records = [], []
for _ in range(runs):
    nn3.netConfig()
    nn3, nn3_record = nn3.train()
    nn3_s.append(nn3)
    nn3_records.append(nn3_record)
nn3_avgR = statModel(nn3_records)

# %%
# mainPlot(21)
# trainPlot(22)
avgPlot(23)

# %%

# tY_pred_confidence = ndt_record.tY_pred_confidence[-1]
# tY_pred_confidence = np.reshape(tY_pred_confidence, (len(tY_pred_confidence), 1))
#
# tY_pred_label = ndt_record.tY_pred_label[-1]
# tY_pred_label = np.reshape(tY_pred_label, (len(tY_pred_label), 1))
# print(np.sum(np.not_equal(tY_pred_label, ndt.d_test.Y)))

# [W0, b0, W1, b1, W2, b2] = ndt.para
# W0 = W0.numpy()
# W1 = W1.numpy()
# W2 = W2.numpy()
# b0 = b0.numpy()
# b1 = b1.numpy()
# b2 = b2.numpy()

# %%
