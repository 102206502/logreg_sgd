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
def logreg_sgd(X, y, alpha = .001, iters = 50000, eps=1e-4):
    # TODO: fill this procedure as an exercise
    n, d = X.shape

    theta = numpy.zeros((d, 1))
    not_converge = True
    k = 0
    lam = 0.001

    while not_converge:
        not_converge = False
        for i in range(n):
            x = X[i, :]
            xT = numpy.transpose([x])
            # print('xT:\n', xT, xT.shape)
            # print('theta shape', theta.shape)
            y_hat = 1/(1 + numpy.exp((numpy.dot(x, theta)*(-1))))
            # print('y_hat:', y_hat)
            func_g = xT*(y_hat - y[i]) + lam
            # print('func_g:\n', func_g)
            
            theta_k = theta.copy()
            theta = theta + alpha*func_g
            k+=1
        for delta in abs(theta-theta_k):
            if delta > eps:
                not_converge = True

        if k > iters:
            print('k > iters!')
            break
    # print('theta:\n', theta)
    return theta

def sigmoid():
    pass

def predict(X, theta, threshold):
    value = 1/(1 + numpy.exp((numpy.dot(X, theta)*(-1))))
    row, col = X.shape
    y_hat = numpy.zeros(row)
    for i, val in enumerate(value):
        if val > threshold:
            y_hat[i] = 1
        else:
            y_hat[i] = 0
    return y_hat

# plot the ROC curve of your prediction
# x aixes: TPR = TP / ( TP + FN )
# y aixes: FPR = FP / ( FP + TN ) 
def plot_roc_curve(X_test, y_true, theta):
    k = 101
    TPR_x = numpy.zeros(k)
    FPR_y = numpy.zeros(k)
    for n in range(k):
        threshold = n/k-1
        y_pred = predict(X_test, theta, threshold)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        TPR_x[n] = tp/(tp+fn)
        FPR_y[n] = fp/(fp+tn)
        print(tn, fp, fn, tp)
        print(TPR_x[n], FPR_y[n])

    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.plot(TPR_x, FPR_y, '-')
    plt.show()


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.8)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    plot_roc_curve(X_test, y_test, theta)


if __name__ == "__main__":
    main(sys.argv)


