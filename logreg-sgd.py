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
    print('X:\n', X)
    print(X.shape)
    print('y:\n', y)
    import pdb; pdb.set_trace()  # breakpoint ee676081 //

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

    while not_converge:
        func_g = numpy.matmul(X.T, (numpy.matmul(X, theta)-y)) + theta
        theta_k = theta
        theta = theta - alpha*func_g
        not_converge = False
        for delta in abs(theta-theta_k):
        	if delta > eps:
        		not_converge = True
        if k > iters:
            print('k > iters!')
            break
        k+=1
    # print('theta:\n', theta)
    return theta


def predict(X, theta):
    return 1/(1 + numpy.exp((numpy.dot(X, theta)*(-1))))

# plot the ROC curve of your prediction
# x aixes: TPR = TP / ( TP + FN )
# y aixes: FPR = FP / ( FP + TN ) 
def plot_roc_curve():
	pass


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    y_hat = predict(X_train_scale, theta)
    print("Linear train R^2: %f" % (sklearn.metrics.r2_score(y_train, y_hat)))
    y_hat = predict(X_test_scale, theta)
    print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, y_hat)))


if __name__ == "__main__":
    main(sys.argv)


