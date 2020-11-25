import tensorflow as tf
from framework import tfNDT
import numpy as np
import matplotlib.pyplot as plt
from framework.ndtFunc import statModel
from framework import plotsFunc

# from sklearn.datasets import fetch_rcv1
# rcv1 = fetch_rcv1()
dataName = 'Reuters'
max_word = 2000
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.reuters.load_data(num_words=max_word)
word_idx = tf.keras.datasets.reuters.get_word_index()
idx_to_word = dict([(value, key) for (key, value) in word_idx.items()])
print(' '.join([idx_to_word.get(x - 3, '?') for x in X_train[0]]))

# ---------------------------------------------------------------------------------
# plt.figure(2)
# plt.hist(Y_train, bins=45)
# label = 4  # binary classification of selected label vs others
# X_train_selected, Y_train_selected = X_train, tfNDT.ndtFunc.each_label(Y_train, label)
# X_test_selected, Y_test_selected = X_test, tfNDT.ndtFunc.each_label(Y_test, label)
# ---------------------------------------------------------------------------------
X_train_selected, Y_train_selected = X_train, Y_train  # multi-classification
X_test_selected, Y_test_selected = X_test, Y_test
# ---------------------------------------------------------------------------------
d_train = tfNDT.DataProcess(X_train_selected, Y_train_selected)
d_test = tfNDT.DataProcess(X_test_selected, Y_test_selected)
d_train.sequenceToMatrix(dimension=max_word)
d_train.oneHotLabel()
d_test.sequenceToMatrix(dimension=max_word)
d_test.oneHotLabel()
runs = 10


# ---------------------------------------------------------------------------------


def mainPlot(i):
    plt.figure(i)
    plt.plot(ndt_record.epoch_range, E_tree_test * np.ones((ndt.epochs + 1, 1)), color='k', label='DT')
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
    plt.savefig('Improved-NDT/main/figure/Reuters_testing.png', format='png', dpi=100)


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
    plt.title('Reuters Testing')
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
    plt.title('Reuters Training')
    plt.legend()
    plt.show()


def ndtComp(i, datasetName):
    plt.figure(i)
    plt.plot(ndt_record.epoch_range, E_tree_test * np.ones((ndt.epochs + 1, 1)), color='tab:blue', label='DT')
    plt.plot(ndt_record.epoch_range, E_treeH_test * np.ones((ndt.epochs + 1, 1)), color='tab:orange',
             label='DT-P')
    plt.plot(ndt_record.epoch_range, E_tree8_test * np.ones((ndt.epochs + 1, 1)), color='tab:green',
             label='DT-8')
    plt.plot(ndt_record.epoch_range, ndt_record.tClf_error, marker='o', color='tab:blue', label='NDT')
    plt.plot(ndtP_record.epoch_range, ndtP_record.tClf_error, marker='o', color='tab:orange', label='NDT-P')
    plt.plot(ndt8_record.epoch_range, ndt8_record.tClf_error, marker='o', color='tab:green', label='NDT-8')

    # plt.ylim(0, 10)
    plt.xlabel('Epochs ')
    plt.ylabel('Classification Error in %')
    plt.title(datasetName+' Testing')
    plt.legend()
    plt.show()
    plt.savefig('Improved-NDT/main/figure/'+datasetName+'_testing_ndtComp.png', format='png', dpi=100)


# ---------------------------------------------------------------------------------

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
ndt.learning_rate = 50
ndt.batch_size = 1000

ndt, ndt_record = ndt.train()
ndt_avgR = ndt_record

# ---------------------------------------------------------------------------------

ndtP = tfNDT.NeuralDecisionTreeClassification()
ndtP.d_train, ndtP.d_test = d_train, d_test
clfP = ndtP.treeConfig(init_frac=0.5)
yP_pred = clfP.predict(ndtP.d_test.X)
yP_pred = yP_pred.reshape((len(yP_pred), 1))
E_treeH_test = np.mean(np.not_equal(yP_pred, ndtP.d_test.Y)) * 100

ndtP.activation1 = tfNDT.r1
ndtP.learning_rate = 50
ndtP.batch_size = 1000
ndtP, ndtP_record = ndtP.train()
# ---------------------------------------------------------------------------------

ndt8 = tfNDT.NeuralDecisionTreeClassification()
ndt8.tree_max_depth = 8
ndt8.d_train, ndt8.d_test = d_train, d_test
clf8 = ndt8.treeConfig()
y8_pred = clf8.predict(ndt8.d_test.X)
y8_pred = y8_pred.reshape((len(y8_pred), 1))
E_tree8_test = np.mean(np.not_equal(y8_pred, ndt8.d_test.Y)) * 100

ndt8.activation1 = tfNDT.r1
ndt8.learning_rate = 50
ndt8.batch_size = 1000
ndt8, ndt8_record = ndt8.train()

ndtComp(34, dataName)
# ---------------------------------------------------------------------------------

nn = tfNDT.NeuralDecisionTreeClassification()
nn.d_train, nn.d_test = d_train, d_test
nn.learning_rate = 50
nn.batch_size = 1000

nn_s, nn_records = [], []
for _ in range(runs):
    nn.Wb = ndt.Wb  # borrow the shape of Wb
    nn.netConfig(option='randN')  # reinitialize Wb using rand normal
    nn, nn_record = nn.train()
    nn_s.append(nn)
    nn_records.append(nn_record)
nn_avgR = statModel(nn_records)

# ---------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------

# mainPlot(31)
# trainPlot(32)
avgPlot(33)
