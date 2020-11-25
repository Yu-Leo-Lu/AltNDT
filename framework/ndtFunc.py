# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:48:47 2020

@author: leona
"""
import numpy as np


def findSubtree(feature, cLeft, cRight, j):
    # Assume feature[j] is decision node
    # cLeft[j] is a integer such that feature[cLeft[j]] is the left child of feature[j]

    # return start and end position (of feature) of both left subtree and right subtree of feature[j]
    # that is, feature[startL] to feature[endL] is the boundry for left subtree of feature[j]
    ll, lr, rl, rr = cLeft[j], cLeft[j], cRight[j], cRight[j]
    while ll >= 0 or lr >= 0 or rl >= 0 or rr >= 0:
        if ll >= 0:
            startL = ll
            ll = cLeft[startL]
        if lr >= 0:
            endL = lr
            lr = cRight[endL]
        if rl >= 0:
            startR = rl
            rl = cLeft[startR]
        if rr >= 0:
            endR = rr
            rr = cRight[endR]
    return startL, endL, startR, endR


def featureToMatIndex(feature):
    # return idx that count decision node and leaf node seperately
    idx = np.zeros(feature.shape, dtype=int)
    i, j = 0, 0
    for l in range(len(feature)):
        if feature[l] >= 0:  # count decision nodes
            idx[l] = i
            i += 1
        else:  # count leaf nodes
            idx[l] = j
            j += 1
    return idx


def genWeights0(feature, threshold, d):
    # d: # of features in the input
    # regressor.tree_ decide to chech if feature[j]<= threshold[j] in decision node
    # if leaf nodes k, then feature[k], threshold[k]<0
    K = int((len(feature)) / 2)  # binary-splitted tree, K decision nodes and K+1 leaf nodes
    W = np.zeros((d, K))
    idx = featureToMatIndex(feature)
    for l in range(len(feature)):
        if feature[l] >= 0:  # check decision node
            W[feature[l], idx[l]] = -1
    b = threshold[feature >= 0]

    # convert to float 32
    # W = np.float32(W)
    # b = np.float32(b)
    return W, b


def genWeights1(feature, cLeft, cRight):
    # cLeft[j] is a integer such that feature[cLeft[j]] is the left child of feature[j]
    # return weights1, from hyp layer to region layer
    K = int((len(feature)) / 2)  # binary-splitted tree, K decision nodes and K+1 leaf nodes
    W = np.zeros((K, K + 1))
    idx = featureToMatIndex(feature)
    for l in range(len(feature)):
        if feature[l] >= 0:  # check decision node
            startL, endL, startR, endR = findSubtree(feature, cLeft, cRight, l)
            W[idx[l], idx[startL]:idx[endL] + 1] = 1  # leaf node, from left  subtree of decision node idx[l], goes to 1
            W[idx[l],
            idx[startR]:idx[endR] + 1] = -1  # leaf node, from right subtree of decision node idx[l], goes to -1

    b = np.zeros(K + 1)
    for j in range(K + 1):
        w = W[:, j]
        b[j] = -(sum(w[w > 0]) - 0.5)

    # convert to float 32
    # W = np.float32(W)
    # b = np.float32(b)
    return W, b


def genWeights2_reg(feature, value):
    valueLeaf = value[feature < 0]
    W = valueLeaf.reshape((valueLeaf.shape[0], valueLeaf.shape[1] * valueLeaf.shape[2]))
    b = np.zeros(W.shape[1])
    # convert to float 32
    # W = np.float32(W)
    return W, b


def genWeights2_clf(feature, value):
    valueLeaf = value[feature < 0]
    W = valueLeaf.reshape((valueLeaf.shape[0], valueLeaf.shape[1] * valueLeaf.shape[2]))
    b = np.zeros(W.shape[1])
    for i in range(W.shape[0]):
        w = -np.ones((W.shape[1]))
        w[W[i, :] == W[i, :].max()] = 1
        W[i, :] = w

    return W, b


# --------------------------------------------------------------
# store training record
class TrainingRecord:
    def __init__(self):
        self.epoch_range = None
        self.tY_pred_label = None
        self.tY_pred_confidence = None
        self.Cost = None
        self.tCost = None
        self.Clf_error = None
        self.tClf_error = None
        self.Gradients = None


# --------------------------------------------------------------
# select label for binary classification
def select_labels(X, Y, labels):
    idx = np.where(Y == labels[0])[0]
    X_selected, Y_selected = X[idx], Y[idx]
    for l in labels[1:]:
        idx = np.where(Y == l)[0]
        x = X[idx]
        y = Y[idx]
        X_selected = np.vstack((X_selected, x))
        Y_selected = np.vstack((Y_selected, y))
    Y_selected = np.where(Y_selected != labels[0], 1, Y_selected)
    return X_selected, Y_selected


# --------------------------------------------------------------
def each_label(Y, label):
    binaryY = Y
    if label != 0:
        binaryY = np.where(binaryY != label, 0, binaryY)
        binaryY = np.where(binaryY == label, 1, binaryY)
    else:
        binaryY = np.where(binaryY != label, -1, binaryY)
        binaryY = np.where(binaryY == label, 1, binaryY)
        binaryY = np.where(binaryY == -1, 0, binaryY)
    return binaryY


# --------------------------------------------------------------
def statModel(mdl_records):
    """
    :param mdl_records: list of iid model records
    :return avg_record: average of all model records in mdl_records, so far only includes
        1. tClf_error: testing classification error
        2. Cost: training L2 cost
    """

    runs = len(mdl_records)
    avg_record = TrainingRecord()
    avg_record.epoch_range = mdl_records[0].epoch_range
    avg_record.tClf_error = [0] * len(mdl_records[0].epoch_range)
    avg_record.Cost = [0] * len(mdl_records[0].epoch_range)

    # sum of all records
    for j in range(len(avg_record.tClf_error)):  # for each epoch:
        for i in range(runs):  # for each run:
            avg_record.tClf_error[j] += mdl_records[i].tClf_error[j]
            avg_record.Cost[j] += mdl_records[i].Cost[j]
    
    # average in # of records
    for j in range(len(avg_record.tClf_error)):
        avg_record.tClf_error[j] /= runs
        avg_record.Cost[j] /= runs
    return avg_record


# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------

# funcs on prev version of ndt
def sd(sY):
    # input sY:   n by 1 categorical vector, with c classes: 0,1,2,...,c-1
    # output dY:  n by c binary sparse matrix with
    #            dY_{ij} = 1 if in sY, ith data is class j, else 0
    if len(sY.shape) == 1:
        sY = sY.reshape(sY.shape[0], 1)

    n = sY.shape[0]
    classes = np.unique(sY)

    Y = np.zeros([n, 1])

    for j in classes:
        col = np.zeros([n, 1])
        col[np.where(sY == j)[0]] = 1
        Y = np.hstack([Y, col])

    Y = Y[:, 1:]
    return Y


def rightChild(feature, j):
    # input:
    # feature = clf.tree_.feature
    # j:    clf.tree_.feature[j]

    # output:
    # j_right: position of right child of feature[j] in feature
    # feature_count: count the # of decision nodes from feature[j] to right child of feature[j], (] half inclusive
    # that is, feature[j_right] = feature[feature>=0][feature_count]
    feature_count = 0
    if feature[j] < 0:
        return -2, feature_count
    if feature[j + 1] < 0:
        return j + 2, feature_count + 1

    j_right = j + 1  # feature[j+1]>0 from previous if code
    feature_count += 1
    leaf_count = 0
    while j_right < len(feature):
        if feature[j_right] >= 0:
            feature_count += 1
            j_right += 1
            continue
        else:
            leaf_count += 1
            j_right += 1
            if feature_count == leaf_count:
                break

    # not sure why I add this
    #    if feature[j_right]<0:
    #        feature_count=0
    return j_right, feature_count


# ---------------------------------------------------------------------------------
def DataToHyp(X, clf):
    # X:     N x P, N data point, P attributes
    # clf:   tree classifier

    # return: hypData, N x H, mapping from data to hyperplane, <= map 1, > map -1
    H = len(clf.tree_.feature[clf.tree_.feature >= 0])
    hypData = np.zeros((X.shape[0], H))
    for i in range(X.shape[0]):
        # ith data
        col = 0  # col counts the feature in clf.tree_.feature (no leaf)
        j = 0
        #        print('---------------------------')
        while j < len(clf.tree_.feature):
            if clf.tree_.feature[j] < 0:
                j += 1
                continue
            #            print('parent', j)
            if X[i][clf.tree_.feature[j]] <= clf.tree_.threshold[j]:
                hypData[i][col] = 1
                j += 1
                #                print('L child',j)
                if clf.tree_.feature[j] < 0:
                    break
                col += 1
            else:
                hypData[i][col] = -1
                j, f_count = rightChild(clf.tree_.feature, j)
                #                print('R child',j)
                if clf.tree_.feature[j] < 0:
                    break
                col += f_count
    return hypData
