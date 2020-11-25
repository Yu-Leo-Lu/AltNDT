# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:53:26 2019

@author: leona
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from framework.tf4 import NeuralDecisionTree
from framework.ndtFunc import sd
#---------------------------------------------------------------------------------
def splot(X,Y, title_str = 'Plot'):
#    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('equal')
    plt.scatter(X[:,0], X[:,1], c = Y.reshape(Y.shape[0],))
    plt.title(title_str)
    
def r(x):
    return tf.keras.activations.relu(x)
def r1(x):
    return tf.keras.activations.relu(x,0,1,0)
def t(x):
    return tf.keras.activations.sigmoid(x)
def s(x):
    return (tf.tanh(x)+1)*0.5

#---------------------------------------------------------------------------------
X = np.load('X_circle.npy')
X = np.float32(X)
sY = np.load('Y_circle.npy')
sY = np.float32(sY)
#---------------------------------------------------------------------------------
n   = 5000 # num of training data
X1  = X[n:,:]
sY1 = sY[n:,:]
X   = X[0:n,:]
sY  = sY[0:n,:]
Y   = sd(sY)
Y1  = sd(sY1)

#X  = preprocessing.scale(X, axis = 0)
#X1 = preprocessing.scale(X1,axis = 0)
#---------------------------------------------------------------------------------
ndt = NeuralDecisionTree(X,sY)
diff = ndt.initNetwork()
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
clf = ndt.clf
Y_tree = ndt.y_tree
Y1_tree = clf.predict(X1)
Y1_tree = Y1_tree.reshape(len(Y1_tree),1)
C_tree_train = np.mean(np.sum((Y-Y_tree)**2, axis = 1))
C_tree_test = np.mean(np.sum((Y1_tree - sY1)**2, axis = 1))
A_tree_train = ndt.tree_accuracy
A_tree_test = np.mean(np.equal(Y1_tree, sY1))
#---------------------------------------------------------------------------------
import graphviz 
from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("my tree") 
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
learning_rate1 = 0.2 #0.2
learning_rate2 = 0.2
learning_rates = 0.2 
learning_ratet = 0.2 
epochs = 10
batch_size = 500
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
A_ndt_train, A_ndt_test, C_ndt_train, C_ndt_test, Y_NDT1      = ndt.ndt(X_test = X1, Y_test = Y1, activation1 = r1, activation2 = s, learning_rate = learning_rate1, epochs = epochs, batch_size = batch_size)
A_nn_train, A_nn_test, C_nn_train, C_nn_test, Y_NN1           = ndt.nn(X_test  = X1, Y_test = Y1, activation = s , learning_rate = learning_rate2, epochs = epochs, batch_size = batch_size)
As_ndt_train, As_ndt_test, Cs_ndt_train, Cs_ndt_test, Y_NDT1s = ndt.ndt(X_test = X1, Y_test = Y1, activation1 = s, activation2 = s,  learning_rate = learning_rates, epochs = epochs, batch_size = batch_size)
At_ndt_train, At_ndt_test, Ct_ndt_train, Ct_ndt_test, Y_NDT1t = ndt.ndt(X_test = X1, Y_test = Y1, activation1 = t, activation2 = t,  learning_rate = learning_ratet, epochs = epochs, batch_size = batch_size)
#---------------------------------------------------------------------------------

#pre_h_layer = ndt.pre_h_layer
#h_layer = ndt.h_layer
#pre_he = ndt.pre_he
#he = ndt.he
#pre_r_layer=ndt.pre_r_layer
#r_layer=ndt.r_layer
#pre_re = ndt.pre_re
#re = ndt.re
#pre_y_e = ndt.pre_y_e
#y_e = ndt.y_e
#---------------------------------------------------------------------------------
#R = 5
#for i in range(1,R):
#    ndt = NeuralDecisionTree(X,sY)
#    diff = ndt.initNetwork()
#    clf = ndt.clf
#    Y_tree = ndt.y_tree
#    Y1_tree = clf.predict(X1)
#    Y1_tree = Y1_tree.reshape(len(Y1_tree),1)
#
#    a_tree_train = ndt.tree_accuracy
#    a_tree_test = np.mean(np.equal(Y1_tree, sY1))
#    a_ndt_train, a_ndt_test, c_ndt_train, c_ndt_test, y_NDT1              = ndt.ndt(X_test = X1, Y_test = Y1, activation1 = r1, activation2 = s, learning_rate = learning_rate1, epochs = epochs, batch_size = batch_size)
#    as_ndt_train, as_ndt_test, cs_ndt_train, cs_ndt_test, y_NDT1s = ndt.ndt(X_test = X1, Y_test = Y1, activation1 = s, activation2 = s,  learning_rate = learning_rates, epochs = epochs, batch_size = batch_size)
#    at_ndt_train, at_ndt_test, ct_ndt_train, ct_ndt_test, y_NDT1t = ndt.ndt(X_test = X1, Y_test = Y1, activation1 = r, activation2 = t,  learning_rate = learning_ratet, epochs = epochs, batch_size = batch_size)
#    a_nn_train, a_nn_test,c_nn_train,c_nn_test, y_NN1                    = ndt.nn (X_test = X1, Y_test = Y1, activation = s, learning_rate = learning_rate2, epochs = epochs, batch_size = batch_size)
#    
#    A_tree_train = np.add(A_tree_train, a_tree_train)
#    A_tree_test  = np.add(A_tree_test, a_tree_test)
#    A_ndt_train  = np.add(A_ndt_train,a_ndt_train)
#    A_ndt_test   = np.add(A_ndt_test,a_ndt_test)
#    As_ndt_train  = np.add(As_ndt_train,as_ndt_train)
#    As_ndt_test   = np.add(As_ndt_test,as_ndt_test)
#    At_ndt_train  = np.add(At_ndt_train,at_ndt_train)
#    At_ndt_test   = np.add(At_ndt_test,at_ndt_test)
#    A_nn_train   = np.add(A_nn_train,a_nn_train)
#    A_nn_test    = np.add(A_nn_test,a_nn_test)
#
#    C_ndt_train  = np.add(C_ndt_train,c_ndt_train)
#    C_ndt_test   = np.add(C_ndt_test,c_ndt_test)
#    Cs_ndt_train  = np.add(Cs_ndt_train,cs_ndt_train)
#    Cs_ndt_test   = np.add(Cs_ndt_test,cs_ndt_test)
#    Ct_ndt_train  = np.add(Ct_ndt_train,ct_ndt_train)
#    Ct_ndt_test   = np.add(Ct_ndt_test,ct_ndt_test)
#    C_nn_train   = np.add(C_nn_train,c_nn_train)
#    C_nn_test    = np.add(C_nn_test,c_nn_test)
#    
#    Y_NDT1 = np.add(Y_NDT1, y_NDT1)
#    Y_NDT1s = np.add(Y_NDT1s, y_NDT1s)
#    Y_NDT1t = np.add(Y_NDT1t, y_NDT1t)
#    Y_NN1 = np.add(Y_NN1, y_NN1)
#
#A_tree_train/=R
#A_tree_test/=R
#A_ndt_train/=R
#A_ndt_test/=R
#As_ndt_train/=R
#As_ndt_test/=R
#At_ndt_train/=R
#At_ndt_test/=R
#A_nn_train/=R
#A_nn_test/=R
#
#C_ndt_train/=R
#C_ndt_test/=R
#Cs_ndt_train/=R
#Cs_ndt_test/=R
#Ct_ndt_train/=R
#Ct_ndt_test/=R
#C_nn_train/=R
#C_nn_test/=R
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#plt.figure(1,figsize = (8,4))
#plt.plot(range(epochs),A_tree_train*np.ones([epochs,1]), linestyle='-.', label = 'DT')
#plt.plot(range(epochs), A_ndt_train, linestyle='-.', marker='o', color='b',label = 'NDT')
#plt.plot(range(epochs), A_nn_train, linestyle='-.', marker='o', color='r',label = 'NN')
#plt.plot(range(epochs), As_ndt_train, linestyle='-.', marker='o', color='g',label = 'NDTs')
#plt.plot(range(epochs), At_ndt_train, linestyle='-.', marker='o', color='c',label = 'NDTt')
#plt.plot(range(epochs),A_tree_test*np.ones([epochs,1]),label = 'DT_test')
#plt.plot(range(epochs),A_ndt_test, marker='o', color='b',label = 'NDT_test')
#plt.plot(range(epochs),A_nn_test, marker='o', color='r',label = 'NN_test')
#plt.plot(range(epochs),As_ndt_test, marker='o', color='g',label = 'NDTs_test')
#plt.plot(range(epochs),At_ndt_test, marker='o', color='c',label = 'NDTt_test')
#plt.xlabel('Epochs ')
#plt.ylabel('L2 Error')
#plt.title('111')
#plt.legend()
#plt.show()
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
plt.figure(1,figsize = (8,4))
plt.plot(range(epochs), A_tree_train*np.ones([epochs,1]),  label = 'DT')
plt.plot(range(epochs), A_ndt_train, marker='x', color='b',label = 'NDT')
plt.plot(range(epochs), A_nn_train,  marker='x', color='r',label = 'NN')
plt.plot(range(epochs), As_ndt_train,  marker='x', color='g',label = 'NDTs')
plt.plot(range(epochs), At_ndt_train, marker='x', color='c',label = 'NDTt')
plt.xlabel('Epochs ')
plt.ylabel('L2 Error')
plt.title('Training')
plt.legend()
plt.savefig('Circle_training.jpeg', format='jpeg', dpi=500)
plt.show()
#---------------------------------------------------------------------------------
plt.figure(1,figsize = (8,4))
plt.plot(range(epochs),A_tree_test*np.ones([epochs,1]),label = 'DT')
plt.plot(range(epochs),A_ndt_test, marker='o', color='b',label = 'NDT')
plt.plot(range(epochs),A_nn_test, marker='o', color='r',label = 'NN')
plt.plot(range(epochs),As_ndt_test, marker='o', color='g',label = 'NDTs')
plt.plot(range(epochs),At_ndt_test, marker='o', color='c',label = 'NDTt')
plt.xlabel('Epochs ')
plt.ylabel('Correct Classification Rate')
plt.title('Testing')
plt.legend()
plt.savefig('Circle_testing.jpeg', format='jpeg', dpi=500)
plt.show()
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
x1 = np.arange(-4, 4, 0.1)
x1 = x1.reshape(len(x1),1)
x2 = np.arange(-4, 4, 0.1)
x2 = x2.reshape(len(x2),1)
xx = np.array(np.meshgrid(x1, x2)).T.reshape(-1,2) 
#---------------------------------------------------------------------------------
_,_,_,_, yy_ndt = ndt.ndt(X_test = xx, Y_test = np.zeros([6400,2]), activation1 = r1, activation2 = s, learning_rate = learning_rate1, epochs = epochs, batch_size = batch_size)

plt.figure(2, figsize = (16,4))
plt.subplot(141)
splot(X1,Y_NDT1[0][:,1], title_str = 'NDT, epoch: 0' )
plt.subplot(142)
splot(X1,Y_NDT1[4][:,1], title_str = 'NDT, epoch: 4' )
plt.subplot(143)
splot(X1,Y_NDT1[9][:,1], title_str = 'NDT, epoch: 9' )
plt.subplot(144)
splot(xx,yy_ndt[9][:,1], title_str = 'NDT mesh, epoch: 9' )
plt.savefig('NDT run and error.jpeg', format='jpeg', dpi=500)
plt.show()

_,_,_,_, yy_ndt = ndt.ndt(X_test = xx, Y_test = np.zeros([6400,2]), activation1 = s, activation2 = s, learning_rate = learning_rate1, epochs = epochs, batch_size = batch_size)

plt.figure(2, figsize = (16,4))
plt.subplot(141)
splot(X1,Y_NDT1s[0][:,1], title_str = 'NDTs, epoch: 0' )
plt.subplot(142)
splot(X1,Y_NDT1s[4][:,1], title_str = 'NDTs, epoch: 4' )
plt.subplot(143)
splot(X1,Y_NDT1s[9][:,1], title_str = 'NDTs, epoch: 9' )
plt.subplot(144)
splot(xx,yy_ndt[9][:,1], title_str = 'NDTs mesh, epoch: 9' )
plt.savefig('NDTs run and error.jpeg', format='jpeg', dpi=500)
plt.show()

_,_,_,_, yy_ndt = ndt.ndt(X_test = xx, Y_test = np.zeros([6400,2]), activation1 = t, activation2 = t, learning_rate = learning_rate1, epochs = epochs, batch_size = batch_size)

plt.figure(2, figsize = (16,4))
plt.subplot(141)
splot(X1,Y_NDT1t[0][:,1], title_str = 'NDTt, epoch: 0' )
plt.subplot(142)
splot(X1,Y_NDT1t[4][:,1], title_str = 'NDTt, epoch: 4' )
plt.subplot(143)
splot(X1,Y_NDT1t[9][:,1], title_str = 'NDTt, epoch: 9' )
plt.subplot(144)
splot(xx,yy_ndt[9][:,1], title_str = 'NDTt mesh, epoch: 9' )
plt.savefig('NDTt run and error.jpeg', format='jpeg', dpi=500)
plt.show()

_,_,_,_, yy_nn = ndt.nn(X_test = xx, Y_test = np.zeros([6400,2]), activation = s, learning_rate = learning_rate2, epochs = epochs, batch_size = batch_size)
plt.figure(3, figsize = (16,4))
plt.subplot(141)
splot(X1,Y_NN1[0][:,1], title_str = 'NN, epoch: 0' )
plt.subplot(142)
splot(X1,Y_NN1[4][:,1], title_str = 'NN, epoch: 4' )
plt.subplot(143)
splot(X1,Y_NN1[9][:,1], title_str = 'NN, epoch: 9' )
plt.subplot(144)
splot(xx,yy_nn[9][:,1], title_str = 'NN mesh, epoch: 9')
plt.savefig('NN run and error.jpeg', format='jpeg', dpi=500)
plt.show()

#k = 0
#splot(X1,Y_NDT1[k][:,1], title_str = "NDT" )
#plt.show()
#
#splot(X1,Y_NN1[k][:,1], title_str = "NN" )
#plt.show()

#splot(X1,Y_NDT1s[k][:,1], title_str = "NDTs" )
#plt.show()
#
#splot(X1,Y_NDT1t[k][:,1], title_str = "NDTt" )
#plt.show()
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
yy_tree = clf.predict(xx)
plt.figure(4, figsize=(16, 4))
plt.subplot(141)
splot(X,Y[:,1], 'Training')
plt.subplot(142)
splot(X1,Y1[:,1], 'Testing')
plt.subplot(143)
splot(X1,Y1_tree, 'Decision Tree')
plt.subplot(144)
splot(xx,yy_tree, 'Decision Tree, mesh')
plt.savefig('Training and Tree.jpeg', format='jpeg', dpi=500)
plt.show()
