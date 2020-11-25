from framework import tfNDT
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# dataName = 'MNIST'
# (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
# # plt.imshow(X_train[1658], cmap='Greys')
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
# Y_train = Y_train.reshape(Y_train.shape[0], 1)
# Y_test = Y_test.reshape(Y_test.shape[0], 1)
# ---------------------------------------------------------------------------------
# dataName = 'Reuters'
# max_word = 2000
# (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.reuters.load_data(num_words=max_word)
# word_idx = tf.keras.datasets.reuters.get_word_index()
# idx_to_word = dict([(value, key) for (key, value) in word_idx.items()])
# print(' '.join([idx_to_word.get(x - 3, '?') for x in X_train[0]]))
# ---------------------------------------------------------------------------------
dataName = 'CIFAR10'
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
# plt.imshow(X_train[1658])
X_train = X_train.reshape(50000, 32 * 32 * 3)  # 50000 training samples, each sample is 32x32x3 RGB image
X_test = X_test.reshape(10000, 32 * 32 * 3)  # 10000 training samples
# ---------------------------------------------------------------------------------

X_train_selected, Y_train_selected = X_train, Y_train  # multi-classification
X_test_selected, Y_test_selected = X_test, Y_test
# ---------------------------------------------------------------------------------
d_train = tfNDT.DataProcess(X_train_selected, Y_train_selected)
d_test = tfNDT.DataProcess(X_test_selected, Y_test_selected)

d_train.preProcessData()
d_test.preProcessData()


# d_train.sequenceToMatrix(dimension=max_word)
# d_train.oneHotLabel()
# d_test.sequenceToMatrix(dimension=max_word)
# d_test.oneHotLabel()

# ---------------------------------------------------------------------------------

def ndtComp(i):
    plt.figure(i)
    # plt.plot(ndt_record.epoch_range, E_tree_test * np.ones((ndt.epochs + 1, 1)), color='tab:blue', label='DT')
    # plt.plot(ndtP10_record.epoch_range, E_treeP10_test * np.ones((ndt.epochs + 1, 1)), color='tab:red',
    #          label='DT-10%')
    # plt.plot(ndtP_record.epoch_range, E_treeP_test * np.ones((ndt.epochs + 1, 1)), color='tab:orange',
    #          label='DT-50%')
    # plt.plot(ndt6_record.epoch_range, E_tree6_test * np.ones((ndt6.epochs + 1, 1)), color='tab:purple',
    #          label='DT-6')
    # plt.plot(ndt8_record.epoch_range, E_tree8_test * np.ones((ndt8.epochs + 1, 1)), color='tab:green',
    #          label='DT-8')
    # plt.plot(ndt9_record.epoch_range, E_tree9_test * np.ones((ndt9.epochs + 1, 1)), color='tab:brown',
    #          label='DT-9')
    plt.plot(ndt_record.epoch_range, ndt_record.tClf_error, marker='o', color='tab:blue', label='NDT')
    plt.plot(ndtP10_record.epoch_range, ndtP10_record.tClf_error, marker='o', color='tab:red', label='NDT-10%')
    plt.plot(ndtP_record.epoch_range, ndtP_record.tClf_error, marker='o', color='tab:orange', label='NDT-50%')
    plt.plot(ndt6_record.epoch_range, ndt6_record.tClf_error, marker='o', color='tab:purple', label='NDT-6')
    plt.plot(ndt8_record.epoch_range, ndt8_record.tClf_error, marker='o', color='tab:green', label='NDT-8')
    plt.plot(ndt9_record.epoch_range, ndt9_record.tClf_error, marker='o', color='tab:brown', label='NDT-9')

    # plt.ylim(0, 10)
    plt.yscale('log')
    plt.xticks(np.arange(ndt.epochs + 1, step=5))
    plt.xlabel('Epochs ')
    plt.ylabel('Classification Error in log%')
    plt.title(dataName + ' Testing ')
    plt.legend()
    plt.show()
    plt.savefig('Improved-NDT/main/figure/' + dataName + '_testing_ndtComp.png', format='png', dpi=100)


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
ndt.learning_rate = 25
ndt.batch_size = 1000

ndt, ndt_record = ndt.train()
ndt_avgR = ndt_record

# ---------------------------------------------------------------------------------

ndtP = tfNDT.NeuralDecisionTreeClassification()
ndtP.d_train, ndtP.d_test = d_train, d_test
clfP = ndtP.treeConfig(init_frac=0.5)
yP_pred = clfP.predict(ndtP.d_test.X)
yP_pred = yP_pred.reshape((len(yP_pred), 1))
E_treeP_test = np.mean(np.not_equal(yP_pred, ndtP.d_test.Y)) * 100

ndtP.activation1 = tfNDT.r1
ndtP.learning_rate = 25
ndtP.batch_size = 1000
ndtP, ndtP_record = ndtP.train()

# ---------------------------------------------------------------------------------

ndtP10 = tfNDT.NeuralDecisionTreeClassification()
ndtP10.d_train, ndtP10.d_test = d_train, d_test
clfP10 = ndtP10.treeConfig(init_frac=0.1)
yP10_pred = clfP10.predict(ndtP10.d_test.X)
yP10_pred = yP10_pred.reshape((len(yP10_pred), 1))
E_treeP10_test = np.mean(np.not_equal(yP10_pred, ndtP10.d_test.Y)) * 100

ndtP10.activation1 = tfNDT.r1
ndtP10.learning_rate = 25
ndtP10.batch_size = 1000
ndtP10, ndtP10_record = ndtP10.train()

# ---------------------------------------------------------------------------------

ndt8 = tfNDT.NeuralDecisionTreeClassification()
ndt8.tree_max_depth = 8
ndt8.d_train, ndt8.d_test = d_train, d_test
clf8 = ndt8.treeConfig()
y8_pred = clf8.predict(ndt8.d_test.X)
y8_pred = y8_pred.reshape((len(y8_pred), 1))
E_tree8_test = np.mean(np.not_equal(y8_pred, ndt8.d_test.Y)) * 100

ndt8.activation1 = tfNDT.r1
ndt8.learning_rate = 25
ndt8.batch_size = 1000
ndt8, ndt8_record = ndt8.train()

# ---------------------------------------------------------------------------------

ndt6 = tfNDT.NeuralDecisionTreeClassification()
ndt6.tree_max_depth = 6
ndt6.d_train, ndt6.d_test = d_train, d_test
clf6 = ndt6.treeConfig()
y6_pred = clf6.predict(ndt6.d_test.X)
y6_pred = y6_pred.reshape((len(y6_pred), 1))
E_tree6_test = np.mean(np.not_equal(y6_pred, ndt6.d_test.Y)) * 100

ndt6.activation1 = tfNDT.r1
ndt6.learning_rate = 25
ndt6.batch_size = 1000
ndt6, ndt6_record = ndt6.train()

# ---------------------------------------------------------------------------------

ndt9 = tfNDT.NeuralDecisionTreeClassification()
ndt9.tree_max_depth = 9
ndt9.d_train, ndt9.d_test = d_train, d_test
clf9 = ndt9.treeConfig()
y9_pred = clf9.predict(ndt9.d_test.X)
y9_pred = y9_pred.reshape((len(y9_pred), 1))
E_tree9_test = np.mean(np.not_equal(y9_pred, ndt9.d_test.Y)) * 100

ndt9.activation1 = tfNDT.r1
ndt9.learning_rate = 25
ndt9.batch_size = 1000
ndt9, ndt9_record = ndt9.train()

ndtComp(14)
