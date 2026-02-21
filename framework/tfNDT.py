# %%
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from framework import ndtFunc
import tensorflow as tf
from sklearn.model_selection import train_test_split

# %%
# some common activation functions
def r(x):
    return tf.keras.activations.relu(x)


def r1(x):
    return tf.keras.activations.relu(x, 0, 1, 0)


def s(x):
    return tf.keras.activations.sigmoid(x)


def t(x):
    return (tf.tanh(x) + 1) * 0.5

# %% DataProcess class, scaling and one-hot encoding

class DataProcess(object):
    def __init__(self, X, Y):
        self.X = X
        if len(Y.shape) == 1:
            self.Y = np.reshape(Y, (len(Y), 1))
        else:
            self.Y = Y
        self.Order = np.arange(self.X.shape[0])
        self.invOrder = np.arange(self.X.shape[0])
        self.Y_oneHot = None

    def shuffleRow(self):
        np.random.shuffle(self.Order)
        for i, each in enumerate(self.Order):
            self.invOrder[each] = i
        self.X = self.X[self.Order, :]
        self.Y = self.Y[self.Order, :]

    def scaling(self):
        self.X = preprocessing.scale(self.X)

    def oneHotLabel(self):
        """
        input Y:   n by 1 categorical vector, with c labels: 0,1,2,...,c-1
        output Y_oneHot:  n by c binary matrix with
        Y_oneHot{ij} = 1 if in Y, ith data is label j, else 0
        """
        if len(self.Y.shape) == 1:
            self.Y = self.Y.reshape(self.Y.shape[0], 1)
        n_row = self.Y.shape[0]
        labels = np.unique(self.Y)
        Y_oneHot = np.zeros([n_row, 1])
        for label in labels:
            col = np.zeros([n_row, 1])
            col[np.where(self.Y == label)[0]] = 1
            Y_oneHot = np.hstack([Y_oneHot, col])
        Y_oneHot = Y_oneHot[:, 1:]
        self.Y_oneHot = Y_oneHot

    def preProcessData(self):
        # self.shuffleRow()
        self.scaling()
        self.oneHotLabel()

    def sequenceToMatrix(self, dimension):
        # for Reuter newswires dataset
        # convert numbered sentence to oneHot
        numSequence = len(self.X)
        X = np.zeros((numSequence, dimension))
        for i in range(numSequence):
            sample_i = self.X[i]
            X[i, sample_i] = 1

        self.X = X

    def fracAllLabels(self, init_frac):
        # for Reuter newswires dataset
        """
        :param: init_frac: float 0~1 inclusive
        :return: X_selected, Y_selected: ndarray
        """
        # select init_frac of total data, base on each class has the same number of data

        numLabels = len(np.unique(self.Y))  # num of labels
        # 0.1 * 2246 / 46
        # only maxNumData for each label can be selected
        maxNumData = (init_frac * len(self.X)) / numLabels + 1
        select_inds = []
        d_ct = {}
        for i in range(len(self.X)):
            d_ct[self.Y[i][0]] = d_ct.get(self.Y[i][0], 0) + 1
            if d_ct[self.Y[i][0]] <= maxNumData:
                select_inds.append(i)
            else:
                continue

        return self.X[select_inds, :], self.Y[select_inds, :]

# %% TuningPara class, tuning parameters for the model
class TuningPara(object):
    def __init__(self):
        self.activation1 = s
        self.activation2 = s
        self.tree_max_depth = 5
        self.epochs = 20
        self.learning_rate = 0.1
        self.batch_size = 1000


class NeuralDecisionTreeRegression:
    def __init__(self):
        self.activation1 = r1
        self.activation2 = s
        self.tree_max_depth = 7
        self.epochs = 5
        self.learning_rate = 0.01
        self.batch_size = 100
        self.Wb = []

    def treeConfig(self, X_train, Y_train):
        X_train_scaled = preprocessing.scale(X_train)
        Y_train_scaled = Y_train

        regressor = tree.DecisionTreeRegressor(max_depth=self.tree_max_depth)
        regressor = regressor.fit(X_train_scaled, Y_train_scaled)
        feature = regressor.tree_.feature
        threshold = regressor.tree_.threshold
        cLeft = regressor.tree_.children_left
        cRight = regressor.tree_.children_right
        value = regressor.tree_.value

        # Call ndtFunc to get weights
        W0, b0 = ndtFunc.genWeights0(feature, threshold, X_train_scaled.shape[1])
        W1, b1 = ndtFunc.genWeights1(feature, cLeft, cRight)
        W2, b2 = ndtFunc.genWeights2_reg(feature, value)
        self.Wb = [W0, b0, W1, b1, W2, b2]

        return regressor

    @tf.function
    def cost(self, X, Y, W0, b0, W1, b1, W2, b2):
        pre_h = tf.add(tf.matmul(X, W0), b0)
        h = self.activation1(pre_h)
        pre_r = tf.add(tf.matmul(h, W1), b1)
        r = self.activation2(pre_r)
        predictions = tf.add(tf.matmul(r, W2), b2)
        loss = tf.reduce_mean(tf.math.squared_difference(Y, predictions))
        return loss, predictions

    def train(self, X_train, Y_train, X_test, Y_test):
        X_train_scaled = preprocessing.scale(X_train)
        Y_train_scaled = Y_train
        X_test_scaled = preprocessing.scale(X_test)
        Y_test_scaled = Y_test

        n = X_train_scaled.shape[0]  # num of obvs

        [W0, b0, W1, b1, W2, b2] = self.Wb
        W0 = tf.Variable(W0, name='W0')
        b0 = tf.Variable(b0, name='b1')
        W1 = tf.Variable(W1, name='W1')
        b1 = tf.Variable(b1, name='b1')
        W2 = tf.Variable(W2, name='W2')
        b2 = tf.Variable(b2, name='b2')
        self.Wb = [W0, b0, W1, b1, W2, b2]

        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        num_batches = int(n / self.batch_size)

        C_train = []
        C_test = []
        Y_test_pred = []

        c_train, _ = self.cost(X_train_scaled, Y_train_scaled, *self.Wb)
        c_test, y_test_pred = self.cost(X_test_scaled, Y_test_scaled, *self.Wb)
        C_train.append(c_train.numpy())
        C_test.append(c_test.numpy())
        Y_test_pred.append(y_test_pred.numpy())

        for epoch in range(self.epochs):
            avg_cost = 0
            for i in range(num_batches):
                batch_x = X_train_scaled[i * self.batch_size:(i + 1) * self.batch_size, ]
                batch_y = Y_train_scaled[i * self.batch_size:(i + 1) * self.batch_size, ]
                with tf.GradientTape() as tape:
                    c, _ = self.cost(batch_x, batch_y, *self.Wb)

                gradients = tape.gradient(c, self.Wb)
                optimizer.apply_gradients(zip(gradients, self.Wb))
                avg_cost += c / num_batches

            c1, y1 = self.cost(X_test_scaled, Y_test_scaled, *self.Wb)
            C_train.append(avg_cost.numpy())
            C_test.append(c1.numpy())
            Y_test_pred.append(y1.numpy())
            tf.print("NDT Epoch:", (epoch + 1), "MSE cost =", "{:.3f}".format(avg_cost))

        training_record = {'epochs': range(self.epochs + 1), 'tYpred': Y_test_pred,
                           'cost': C_train, 'tcost': C_test}
        return self, training_record

    def predict(self, X, Y):
        X_scaled = preprocessing.scale(X)
        Y_scaled = Y
        loss, predictions = self.cost(X_scaled, Y_scaled, *self.Wb)
        return loss.numpy(), predictions.numpy()


class NeuralDecisionTreeClassification(TuningPara):

    def __init__(self):
        self.Wb = []
        self.d_train = None
        self.d_test = None
        super().__init__()

    def dataConfig(self, X_train, Y_train, X_test, Y_test):
        self.d_train = DataProcess(X_train, Y_train)
        self.d_train.preProcessData()
        self.d_test = DataProcess(X_test, Y_test)
        self.d_test.preProcessData()

    def treeConfig(self, init_frac=1):
        """
        initialize Wb using tree
        """
        X_train = self.d_train.X
        Y_train = self.d_train.Y
        # X_train = X_train[0:int(init_frac * X_train.shape[0]), :]
        # Y_train = Y_train[0:int(init_frac * Y_train.shape[0]), :]
        if init_frac < 1:
            X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=1-init_frac)

        classifier = tree.DecisionTreeClassifier(max_depth=self.tree_max_depth)
        classifier = classifier.fit(X_train, Y_train)
        feature = classifier.tree_.feature
        threshold = classifier.tree_.threshold
        cLeft = classifier.tree_.children_left
        cRight = classifier.tree_.children_right
        value = classifier.tree_.value

        # Call ndtFunc to get weights
        W0, b0 = ndtFunc.genWeights0(feature, threshold, X_train.shape[1])
        W1, b1 = ndtFunc.genWeights1(feature, cLeft, cRight)
        W2, b2 = ndtFunc.genWeights2_clf(feature, value)

        W0 = tf.Variable(W0, name='W0')
        b0 = tf.Variable(b0, name='b1')
        W1 = tf.Variable(W1, name='W1')
        b1 = tf.Variable(b1, name='b1')
        W2 = tf.Variable(W2, name='W2')
        b2 = tf.Variable(b2, name='b2')
        self.Wb = [W0, b0, W1, b1, W2, b2]
        return classifier

    def netConfig(self, option='randN'):
        """
        initialize Wb using random normal
        with options of double or half the amount of neurons
        """
        [W0, b0, W1, b1, W2, b2] = self.Wb
        K = W0.shape[1]  # of decision nodes
        if option == 'randN':
            W0 = tf.Variable(tf.random.normal(W0.shape, stddev=1, dtype='float64'), name='W0')
            b0 = tf.Variable(tf.random.normal(b0.shape, stddev=1, dtype='float64'), name='b0')
            W1 = tf.Variable(tf.random.normal(W1.shape, stddev=1, dtype='float64'), name='W1')
            b1 = tf.Variable(tf.random.normal(b1.shape, stddev=1, dtype='float64'), name='b1')
            W2 = tf.Variable(tf.random.normal(W2.shape, stddev=1, dtype='float64'), name='W2')
            b2 = tf.Variable(tf.random.normal(b2.shape, stddev=1, dtype='float64'), name='b2')
        elif option == 'double':
            W0 = tf.Variable(tf.random.normal((W0.shape[0], 2 * K), stddev=1, dtype='float64'), name='W0')
            b0 = tf.Variable(tf.random.normal((2 * K,), stddev=1, dtype='float64'), name='b0')
            W1 = tf.Variable(tf.random.normal((2 * K, 2 * K + 2), stddev=1, dtype='float64'), name='W1')
            b1 = tf.Variable(tf.random.normal((2 * K + 2,), stddev=1, dtype='float64'), name='b1')
            W2 = tf.Variable(tf.random.normal((2 * K + 2, W2.shape[1]), stddev=1, dtype='float64'), name='W2')
            b2 = tf.Variable(tf.random.normal(b2.shape, stddev=1, dtype='float64'), name='b2')
        elif option == 'half':
            W0 = tf.Variable(tf.random.normal((W0.shape[0], int(K / 2)), stddev=1, dtype='float64'), name='W0')
            b0 = tf.Variable(tf.random.normal((int(K / 2),), stddev=1, dtype='float64'), name='b0')
            W1 = tf.Variable(tf.random.normal((int(K / 2), int((K + 1) / 2)), stddev=1, dtype='float64'), name='W1')
            b1 = tf.Variable(tf.random.normal((int((K + 1) / 2),), stddev=1, dtype='float64'), name='b1')
            W2 = tf.Variable(tf.random.normal((int((K + 1) / 2), W2.shape[1]), stddev=1, dtype='float64'), name='W2')
            b2 = tf.Variable(tf.random.normal(b2.shape, stddev=1, dtype='float64'), name='b2')

        # elif option == 'one_layer':
        #
        # elif option == 'three_layers':

        self.Wb = [W0, b0, W1, b1, W2, b2]
        return

    @tf.function
    def cost(self, X, Y_oneHot, W0, b0, W1, b1, W2, b2):
        layer_1 = self.activation1(tf.add(tf.matmul(X, W0), b0))
        layer_2 = self.activation2(tf.add(tf.matmul(layer_1, W1), b1))
        Y_oneHot_pred = tf.nn.softmax(tf.add(tf.matmul(layer_2, W2), b2))
        c = tf.reduce_mean(tf.math.squared_difference(Y_oneHot, Y_oneHot_pred))

        Y_pred = tf.argmax(Y_oneHot_pred, 1)
        Y_pred_confidence = Y_oneHot_pred[:, 1]
        isWrong = tf.not_equal(tf.argmax(Y_oneHot, 1), tf.argmax(Y_oneHot_pred, 1))
        clf_error = tf.reduce_mean(tf.cast(isWrong, tf.float64)) * 100
        return c, [Y_pred, Y_pred_confidence], clf_error

    def train(self):
        X_train = self.d_train.X
        Y_oneHot = self.d_train.Y_oneHot
        X_test = self.d_test.X
        tY_oneHot = self.d_test.Y_oneHot
        n = X_train.shape[0]  # num of obvs

        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        num_batches = int(n / self.batch_size)

        C_train = []
        C_test = []
        Y_test_pred_label = []
        Y_test_pred_confidence = []
        Clf_error_train = []
        Clf_error_test = []
        Gradients = []

        c_train, _, clf_error_train = self.cost(X_train, Y_oneHot, *self.Wb)
        c_test, y_test_pred, clf_error_test = self.cost(X_test, tY_oneHot, *self.Wb)
        C_train.append(c_train.numpy())
        Clf_error_train.append(clf_error_train.numpy())
        C_test.append(c_test.numpy())
        Clf_error_test.append(clf_error_test.numpy())
        Y_test_pred_label.append(y_test_pred[0].numpy())
        Y_test_pred_confidence.append(y_test_pred[1].numpy())

        tf.print("\n--Training--\n")
        for epoch in range(self.epochs):
            avg_cost = 0
            for i in range(num_batches):
                batch_x = X_train[i * self.batch_size: (i + 1) * self.batch_size, ]
                batch_y = Y_oneHot[i * self.batch_size: (i + 1) * self.batch_size, ]
                with tf.GradientTape() as tape:
                    c, _, _ = self.cost(batch_x, batch_y, *self.Wb)

                gradients = tape.gradient(c, self.Wb)
                optimizer.apply_gradients(zip(gradients, self.Wb))
                avg_cost += c / num_batches

            Gradients.append(gradients)
            c_train, _, e_train = self.cost(X_train, Y_oneHot, *self.Wb)
            c1, y1, e1 = self.cost(X_test, tY_oneHot, *self.Wb)
            C_train.append(c_train.numpy())
            C_test.append(c1.numpy())
            Clf_error_train.append(e_train.numpy())
            Clf_error_test.append(e1.numpy())
            Y_test_pred_label.append(y1[0].numpy())
            Y_test_pred_confidence.append(y1[1].numpy())
            tf.print("Epoch", (epoch + 1), "Cost =", "{:.3f}".format(avg_cost))

        tr = ndtFunc.TrainingRecord()
        tr.epoch_range = range(self.epochs + 1)
        tr.tY_pred_label = Y_test_pred_label
        tr.tY_pred_confidence = Y_test_pred_confidence
        tr.Cost, tr.tCost = C_train, C_test
        tr.Clf_error, tr.tClf_error = Clf_error_train, Clf_error_test
        tr.Gradients = Gradients
        return self, tr

    def statsTrain(self, runs=3, init_type='tree', option='randN', init_frac=1):
        """

        :param runs:
        :param init_type: tree (NDT) or standard(else, i.e. NN, NN3, NN-1, etc)
        :param option: randN (NN, NN-1, NN-3), double(NN-D), half(NN-H)
        :param init_frac: init portion of tree in NDT

        :return: mdls, mdl_records: list of iid models, model records
        avg_records: average of all model records in mdl_records, so far only includes
        1. tClf_error: testing classification error
        2. Cost: training L2 cost
        """
        mdls, mdl_records = [], []
        if init_type == 'tree':
            for _ in range(runs):
                self.treeConfig(init_frac=init_frac)
                mdl, mdl_record = self.train()
                mdls.append(mdl)
                mdl_records.append(mdl_record)

        elif init_type == 'standard':
            for _ in range(runs):
                self.netConfig(option=option)
                mdl, mdl_record = self.train()
                mdls.append(mdl)
                mdl_records.append(mdl_record)
        else:
            print('Select type: tree/standard')

        avg_record = ndtFunc.TrainingRecord()
        avg_record.epoch_range = mdl_records[0].epoch_range
        avg_record.tClf_error = [0] * len(mdl_records[0].epoch_range)
        avg_record.Cost = [0] * len(mdl_records[0].epoch_range)

        # sum of all records
        for i in range(runs):
            for j in range(len(avg_record.tClf_error)):
                avg_record.tClf_error[j] += mdl_records[i].tClf_error[j]
                avg_record.Cost[j] += mdl_records[i].Cost[j]

        # average in # of records
        for j in range(len(avg_record.tClf_error)):
            avg_record.tClf_error[j] /= runs
            avg_record.Cost[j] /= runs

        return mdls, mdl_records, avg_record

    def predict(self, X, Y):
        d_pred = DataProcess(X, Y)
        X_scaled, Y_oneHot = d_pred.X, d_pred.Y
        c, [y_pred, y_pred_confidence], clf_error = self.cost(X_scaled, Y_oneHot, *self.Wb)
        return c.numpy(), [y_pred.numpy(), y_pred_confidence.numpy()], clf_error.numpy()


class OneLayersNetworkClassification(NeuralDecisionTreeClassification):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

    def netConfig(self, option='randN'):
        W0 = tf.Variable(tf.random.normal((self.d_train.X.shape[1], self.num_neurons), stddev=1, dtype='float64'),
                         name='W0')
        b0 = tf.Variable(tf.random.normal((self.num_neurons,), stddev=1, dtype='float64'), name='b0')

        W1 = tf.Variable(
            tf.random.normal((self.num_neurons, self.d_train.Y_oneHot.shape[1]), stddev=1, dtype='float64'), name='W1')
        b1 = tf.Variable(tf.random.normal((self.d_train.Y_oneHot.shape[1],), stddev=1, dtype='float64'), name='b1')
        self.Wb = [W0, b0, W1, b1]
        return 'Weight initiated'

    @tf.function
    def cost(self, X, Y_oneHot, W0, b0, W1, b1):
        layer_1 = self.activation2(tf.add(tf.matmul(X, W0), b0))
        Y_oneHot_pred = tf.nn.softmax(tf.add(tf.matmul(layer_1, W1), b1))
        c = tf.reduce_mean(tf.math.squared_difference(Y_oneHot, Y_oneHot_pred))

        Y_pred = tf.argmax(Y_oneHot_pred, 1)
        Y_pred_confidence = Y_oneHot_pred[:, 1]
        isWrong = tf.not_equal(tf.argmax(Y_oneHot, 1), tf.argmax(Y_oneHot_pred, 1))
        clf_error = tf.reduce_mean(tf.cast(isWrong, tf.float64)) * 100
        return c, [Y_pred, Y_pred_confidence], clf_error


class ThreeLayersNetworkClassification(NeuralDecisionTreeClassification):
    def __init__(self, num_neurons_list):
        super().__init__()
        self.num_neurons_list = num_neurons_list

    def netConfig(self, option='randN'):
        K1, K2, K3 = self.num_neurons_list

        W0 = tf.Variable(tf.random.normal((self.d_train.X.shape[1], K1), stddev=1, dtype='float64'), name='W0')
        b0 = tf.Variable(tf.random.normal((K1,), stddev=1, dtype='float64'), name='b0')

        W1 = tf.Variable(tf.random.normal((K1, K2), stddev=1, dtype='float64'), name='W1')
        b1 = tf.Variable(tf.random.normal((K2,), stddev=1, dtype='float64'), name='b1')

        W2 = tf.Variable(tf.random.normal((K2, K3), stddev=1, dtype='float64'), name='W2')
        b2 = tf.Variable(tf.random.normal((K3,), stddev=1, dtype='float64'), name='b2')

        W3 = tf.Variable(tf.random.normal((K3, self.d_train.Y_oneHot.shape[1]), stddev=1, dtype='float64'), name='W3')
        b3 = tf.Variable(tf.random.normal((self.d_train.Y_oneHot.shape[1],), stddev=1, dtype='float64'), name='b3')

        self.Wb = [W0, b0, W1, b1, W2, b2, W3, b3]
        return 'Weight initiated'

    @tf.function
    def cost(self, X, Y_oneHot, W0, b0, W1, b1, W2, b2, W3, b3):
        layer_1 = self.activation2(tf.add(tf.matmul(X, W0), b0))
        layer_2 = self.activation2(tf.add(tf.matmul(layer_1, W1), b1))
        layer_3 = self.activation2(tf.add(tf.matmul(layer_2, W2), b2))
        Y_oneHot_pred = tf.nn.softmax(tf.add(tf.matmul(layer_3, W3), b3))

        c = tf.reduce_mean(tf.math.squared_difference(Y_oneHot, Y_oneHot_pred))

        Y_pred = tf.argmax(Y_oneHot_pred, 1)
        Y_pred_confidence = Y_oneHot_pred[:, 1]
        isWrong = tf.not_equal(tf.argmax(Y_oneHot, 1), tf.argmax(Y_oneHot_pred, 1))
        clf_error = tf.reduce_mean(tf.cast(isWrong, tf.float64)) * 100
        return c, [Y_pred, Y_pred_confidence], clf_error


class NDTMulti(TuningPara):
    def __init__(self, X_train, Y_train, X_test, Y_test):
        super().__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.model_dict = {}
        self.record_dict = {}
        self.confidence_mat = []
        self.tY_pred_label = []
        self.tClf_error = []
        self.labels = np.unique(self.Y_train)

    def ovaTraining(self):
        for label in self.labels:
            X_train, Y_train = self.X_train, ndtFunc.each_label(self.Y_train, label)
            X_test, Y_test = self.X_test, ndtFunc.each_label(self.Y_test, label)
            ndt = NeuralDecisionTreeClassification()
            ndt.dataConfig(X_train, Y_train, X_test, Y_test)
            ndt.treeConfig()
            ndt.epochs, ndt.tree_max_depth = self.epochs, self.tree_max_depth
            ndt.activation1, ndt.activation2 = self.activation1, self.activation2
            ndt.learning_rate, ndt.batch_size = self.learning_rate, self.batch_size

            ndt, ndt_record = ndt.train()
            self.model_dict[label] = ndt
            self.record_dict[label] = ndt_record
            print("End of training label: ", label)

    def multiLabels(self):
        confi_mat = self.record_dict[self.labels[0]].tY_pred_confidence
        epochs = len(confi_mat)
        for label in self.labels[1:]:
            tY_pred_confi_each_label = self.record_dict[self.labels[label]].tY_pred_confidence
            for epoch in range(epochs):
                confi_mat[epoch] = np.vstack([confi_mat[epoch], tY_pred_confi_each_label[epoch]])

        self.confidence_mat = confi_mat

        for epoch in range(epochs):
            tY_pred_each_epoch = np.argmax(confi_mat[epoch], 0)
            tY = self.Y_test.reshape(len(tY_pred_each_epoch))
            tClf_error_each_epoch = np.mean(np.not_equal(tY_pred_each_epoch, tY))
            self.tClf_error.append(tClf_error_each_epoch)
            self.tY_pred_label.append(tY_pred_each_epoch)
