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
        ############(for testing)###################
        y_pdt[k] = y_hat
        y_delta[k] = abs(y_hat - y[i])
        y_true_iters[k] = y[i]
        ############(for testing)###################
        theta_k = theta.copy()
        theta = theta - alpha*func_g
        # print('theta shape:', theta.shape)
        not_converge = True
        for delta in abs(theta-theta_k):
            if delta > eps:
                converge = False
                break

    '''
    y_i = numpy.arange(k+1)
    plt.xlabel('iter')
    plt.ylabel('y_hat')
    print(y_i.shape)
    print(y_pdt.shape)
    x_1, y_1, x_0, y_0 = split_0_1(y_i, y_pdt, y_true_iters)
    plt.scatter(x_0, y_0, marker='x', color='red')
    plt.scatter(x_1, y_1, marker='o', color='blue')
    plt.legend(loc='upper left')
    plt.show()
    '''
    # print('theta:\n', theta)
    return theta

'''for debugging, split data to 'y_true = 0' part and 'y_true = 1' part'''
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

def predict(X, y_true, y_scores, threshold):
    n = X.shape[0]
    
    '''x_axis = numpy.arange(len(y_scores))
    print(x_axis.shape)
    print(y_scores.shape)
    # plt.scatter(x_axis, y_true-y_scores, color='blue')
    for item in y_true-y_scores:
        if item < -1e-1:
            print('FP!')
        else:
            print('no')
    # plt.scatter(x_axis, y_scores, color='blue')
    # plt.scatter(x_axis, y_true, color='red', marker='x')
    # plt.show()
    '''

    y_predict = numpy.zeros(n)


    for i, score in enumerate(y_scores):
        if score > threshold:
            y_predict[i] = 1
            # print('score, threshold', score, threshold)
            # print('true score', y_true[i])
        else:
            y_predict[i] = 0
    FP, TP, FN, TN = confusion_matrix_values(y_predict, y_true)
    # print('FP, TP, FN, TN')
    # print(FP, TP, FN, TN)
    FPR = FP/N
    TPR = TP/P
    print('FPR, TPR', FPR, TPR)
    return y_predict, FPR, TPR

def confusion_matrix_values(y_predict, y_true):
    FP = .0
    TP = .0
    FN = .0
    TN = .0
    for i in range(len(y_predict)):
        if abs(y_predict[i]-y_true[i]) < 1e-6:# == True part
            # print('True part')
            if y_true[i] >= 1:
                TP+=1
            else:
                TN+=1
                # print(y_true[i])
        else:# False part
            # print('False part')
            if y_true[i] <= 0:
                FP+=1
                # print('i=', i)
            else:
                FN+=1
    assert(FP+TP+FN+TN == len(y_predict))
    return FP, TP, FN, TN 

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
    # plt.scatter(FPR_x, TPR_y, marker='o', color='blue')
    plt.show()

def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.8)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)

    scores = sigmoid(X_test, theta)
    # fpr, tpr, thresholds = roc_curve(y_test, scores)
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.plot(fpr, tpr)
    # plt.show()
    plot_roc_curve(X_train, y_train, theta)


if __name__ == "__main__":
    main(sys.argv)


