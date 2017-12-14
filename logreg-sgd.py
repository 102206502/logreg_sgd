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
    print('the persentage of 1:',sum(y)/len(y))
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

# logreg_sgd with L1 regularization
def logreg_sgd(X, y, alpha = .001, iters = 100000, eps=1e-4):

    n, d = X.shape
    theta = numpy.zeros((d, 1))
    k = 0
    lam = 0.001
    not_converge = True
    y_pdt = numpy.zeros(iters)
    y_true_iters = numpy.zeros(iters)
    y_delta = numpy.zeros(iters)

    for k in range(iters):
        if not not_converge:
            break
        i = k%n
        x = X[i, :]
        xT = numpy.transpose([x])
        y_hat = sigmoid(x, theta)
        # print('y_hat:', y_hat)
        beta = de_norm1(theta)

        func_g = (y_hat - y[i])*xT + lam*beta
        theta_k = theta.copy()
        theta = theta - alpha*func_g
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

# plot the ROC curve of your prediction
# x aixes: TPR = TP / ( TP + FN )
# y aixes: FPR = FP / ( FP + TN ) 
def plot_roc_curve(X_test, y_true, theta):
    scores = numpy.squeeze(sigmoid(X_test, theta))
    scores_decent_ord = numpy.argsort(scores)[::-1]
    scores = scores[scores_decent_ord]
    y_true = y_true[scores_decent_ord]
    
    FPR_x = numpy.zeros(len(y_true)+1)
    TPR_y = numpy.zeros(len(y_true)+1)
    P = sum(y_true)
    N = len(y_true) - P
    FP = .0
    TP = .0
    for i, target in enumerate(y_true):
        if target >= 1:
            TP+=1
        else:
            FP+=1
        FPR_x[i] = FP/N
        TPR_y[i] = TP/P

    FPR_x[-1] = .0
    TPR_y[-1] = .0
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(FPR_x, TPR_y)
    for i in range(len(FPR_x)):
        print(FPR_x[i], ',', TPR_y[i])
    # plt.scatter(FPR_x, TPR_y, marker='o', color='blue')
    plt.show()

def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.8)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)
    theta = logreg_sgd(X_train_scale, y_train)
    scores = sigmoid(X_test, theta)
    plot_roc_curve(X_train, y_train, theta)


if __name__ == "__main__":
    main(sys.argv)


