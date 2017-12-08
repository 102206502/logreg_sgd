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
        y_pdt[k] = y_hat
        y_delta[k] = abs(y_hat - y[i])
        y_true_iters[k] = y[i]
        
        theta_k = theta.copy()
        theta = theta - alpha*func_g
        # print('theta shape:', theta.shape)
        not_converge = True
        for delta in abs(theta-theta_k):
            if delta > eps:
                converge = False
                break

    '''y_i = numpy.arange(k+1)
                plt.xlabel('y (i)')
                plt.ylabel('y_hat')
                print(y_i.shape)
                print(y_pdt.shape)
                x_1, y_1, x_0, y_0 = split_0_1(y_i, y_pdt, y_true_iters)
                plt.scatter(x_1, y_1, marker='o', color='blue')
                plt.scatter(x_0, y_0, marker='x', color='red')
                plt.show()'''
    # print('theta:\n', theta)
    return theta

'''for debugging, split an array to '0' part and '1' part'''
def split_0_1(x_axis, y_axis, y_true):
    num_1 = int(sum(y_true))
    print('float, int', sum(y_true), num_1)
    num_0 = len(y_true)-num_1
    x_1 = numpy.zeros(num_1)
    y_1 = numpy.zeros(num_1)
    x_0 = numpy.zeros(num_0)
    y_0 = numpy.zeros(num_0)
    i_1 = 0 # index of _1 list
    i_0 = 0 # index of _0 list

    for i in range(len(x_axis)):
        if y_true[i] == 1:
            x_1[i_1] = x_axis[i]
            y_1[i_1] = y_axis[i]
            i_1+=1
        else:
            x_0[i_0] = x_axis[i]
            y_0[i_0] = y_axis[i]
            i_0+=1

    return x_1, y_1, x_0, y_0

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
    n = X.shape[0]
    value = numpy.squeeze(sigmoid(X, theta))
    
    '''x_axis = numpy.arange(len(value))
    print(x_axis.shape)
    print(value.shape)
    # plt.scatter(x_axis, y_true-value, color='blue')
    for item in y_true-value:
        if item < -1e-1:
            print('FP!')
        else:
            print('no')
    # plt.scatter(x_axis, value, color='blue')
    # plt.scatter(x_axis, y_true, color='red', marker='x')
    # plt.show()
    '''

    # print('value:\n', value)
    y_predict = numpy.zeros(n)
    P = sum(y_true)
    N = n - P

    FP = 0.0
    TP = 0.0

    for i, val in enumerate(value):
        if val > threshold:
            y_predict[i] = 1
            # if threshold > 0.3:
            #     print('val, threshold', val, threshold)
            #     print('true val', y_true[i])
        else:
            y_predict[i] = 0

    for i in range(len(y_predict)):
        if abs(y_predict[i]-y_true[i]) < 1e-6:# == True part
            # print('True part')
            if y_true[i] >= 1:
                TP+=1
                # print(y_true[i])
        else:# False part
            # print('False part')
            if y_true[i] < y_predict[i]:
                FP+=1
                

    # print('fp, tp', FP, TP)
    FPR = FP/N
    TPR = TP/P
    # print('FPR, TPR', FPR, TPR)
    return y_predict, FPR, TPR

# plot the ROC curve of your prediction
# x aixes: TPR = TP / ( TP + FN )
# y aixes: FPR = FP / ( FP + TN ) 
def plot_roc_curve(X_test, y_true, theta):
    k = 31
    FPR_x = numpy.zeros(k)
    TPR_y = numpy.zeros(k)

    for n in range(k):
        threshold = n/(k-1)
        # print('threshold=', threshold)
        y_pred, FPR, TPR = predict(X_test, y_true, theta, threshold)
        FPR_x[n] = FPR
        TPR_y[n] = TPR

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(FPR_x, TPR_y)
    # plt.scatter(FPR_x, TPR_y, marker='o', color='blue')
    plt.show()

def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.8)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)

    scores = sigmoid(X_test, theta)
    # fpr, tpr, thresholds = roc_curve(y_test, scores)
    # plt.plot(fpr, tpr)
    plt.show()
    plot_roc_curve(X_train, y_train, theta)


if __name__ == "__main__":
    main(sys.argv)


