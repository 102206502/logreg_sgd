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


def load_train_test_data(train_ratio=.8):
    data = pandas.read_csv('./HTRU2/HTRU_2.csv', header=None)

    X = data.iloc[:,:8]
    X = numpy.concatenate((numpy.ones((len(X), 1)), X), axis=1)
    y = data.iloc[:,8]
    # print('X:\n', X)
    # print(X.shape)
    # print('y:\n', y)
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
    not_converge = True
    k = 0
    lam = 0

    while not_converge:
        not_converge = False
        for i in range(n):
            x = X[i, :]
            xT = numpy.transpose([x])
            print('xT:\n', xT)
            func_g = numpy.matmul(xT, (sigmoid(x, theta)-y[i+1]))
            import pdb; pdb.set_trace()  # breakpoint 81437e0a //
            
            theta_k = theta
            theta = theta - alpha*func_g
        for delta in abs(theta-theta_k):
            if delta > eps:
                not_converge = True
        if k > iters:
            print('k > iters!')
            break
        k+=1
    # print('theta:\n', theta)
    return theta

def sigmoid(X, theta):
    y = 1/(1 + numpy.exp((numpy.dot(X, theta)*(-1))))
    print(y.shape)
    return y

def predict(X, theta):
    value = sigmoid(X, theta)
    col, row = X.shape
    y_hat = numpy.array(row).zeros()
    for i, val in enumerate(value):
        if val > 0.5:
            y_hat[i] = 1
        else:
            y_hat[i] = 0
    return y_hat

# plot the ROC curve of your prediction
# x aixes: TPR = TP / ( TP + FN )
# y aixes: FPR = FP / ( FP + TN ) 
def plot_roc_curve(X_test, y_true, theta):
    y_pred = predict(X_test, theta)
    con_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    print(con_matrix)


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    plot_roc_curve(X_test, y_test, theta)


if __name__ == "__main__":
    main(sys.argv)


