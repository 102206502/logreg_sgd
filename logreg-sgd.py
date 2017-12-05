#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def load_train_test_data(train_ratio=.8):
    data = pandas.read_csv('./HTRU2/HTRU_2.csv', header=None)

    X = data.iloc[:,:8]
    X = numpy.concatenate((numpy.ones((len(X), 1)), X), axis=1)
    y = data.iloc[:,8]
    y = numpy.array(y)
    # print('X:\n', X)
    # print(X.shape)
    # print('y:\n', y)
    # print('load_train:\ny shape:', y.shape)
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

# logreg_sgd with L1 regularization
def logreg_sgd(X, y, alpha = .001, iters = 100000, eps=1e-4):
    # TODO: fill this procedure as an exercise
    n, d = X.shape
    theta = numpy.zeros((d, 1))
    k = 0
    lam = 0.001
    not_converge = True

    for k in range(iters):
        if not not_converge:
            break
        i = k%n
        x = X[i, :]
        xT = numpy.transpose([x])
        y_hat = sigmoid(x, theta)
        # print('y_hat:', y_hat)
        beta = de_norm1(theta)

        func_g = (y[i] - y_hat)*xT + lam*beta
        if i > n-20:
            print('y_hat:', y_hat, 'y[i]:', y[i])
            print('y_hat - y[i]:', y_hat - y[i])
        
        theta_k = theta.copy()
        theta = theta + alpha*func_g
        # print('theta shape:', theta.shape)
        not_converge = True
        for delta in abs(theta-theta_k):
            if delta > eps:
                converge = False
                break

    # print('theta:\n', theta)
    return theta
def de_norm1(theta):
    d, _ = theta.shape
    beta = numpy.zeros((d, 1))

    for i in range(d):
        if theta[i,0] < 0:
            beta[i,0] = -1
        elif theta[i,0] > 0:
            beta[i,0] = 1
    return beta

def sigmoid(X, theta):
    z = numpy.dot(X, theta)
    value = 1.0/(1.0 + numpy.exp(-z))
    return value

def predict(X, y_true, theta, threshold):
    value = sigmoid(X, theta)
    # print('value:\n', value)
    row, col = X.shape
    y_hat = numpy.zeros(row)
    P = sum(y_true)
    N = row - P
    FP = 0.0
    TP = 0.0
    print('P =', P)

    for i, val in enumerate(value):
        if val > threshold:
            # print('> threshold', val)
            y_hat[i] = 1
            TP+=1
        else:
            y_hat[i] = 0
        if y_true[i] != y_hat[i]:
            if y_true[i] == 0:
                FP+=1
    FPR = FP/N
    TPR = TP/P
    return y_hat, FPR, TPR

# plot the ROC curve of your prediction
# x aixes: TPR = TP / ( TP + FN )
# y aixes: FPR = FP / ( FP + TN ) 
def plot_roc_curve(X_test, y_true, theta):
    k = 51
    FPR_x = numpy.zeros(k)
    TPR_y = numpy.zeros(k)

    for n in range(k):
        threshold = n/(k-1)
        # print('threshold=', threshold)
        y_pred, FPR, TPR = predict(X_test, y_true, theta, threshold)
        # tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        # FPR_x[n] = fp/(fp+tn)
        # TPR_y[n] = tp/(tp+fn)
        FPR_x[n] = FPR
        TPR_y[n] = TPR
        # print(tn, fp, fn, tp)
        print(FPR_x[n], TPR_y[n])

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(FPR_x, TPR_y, '-')
    # plt.scatter(FPR_x, TPR_y, marker='o', color='blue')
    plt.show()

def plot_sigmoid(X, theta, y_true=None):
    n, d = X.shape
    color = ['red', 'blue', 'yellow', 'green', 'black', 'gray', 'orange', 'purple', 'c']
    # print(d)
    y_pred = numpy.zeros((n, n))
    
    for i in range(d):
        # xT = numpy.transpose([X[:, i]])
        # y_pred[i] = sigmoid(xT, theta[i])
        # print(y_pred[i].shape)
        # print(X[:, i].shape)
        plt.scatter(X[:, i], y_true, marker='o', color=color[i])

    plt.show()

def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.8)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    print('theta:\n', theta)
    plot_roc_curve(X_test, y_test, theta)
    # plot_sigmoid(X_test, theta, y_true=y_test)


if __name__ == "__main__":
    main(sys.argv)


