import numpy as np
import matplotlib.pyplot as plt


def r_squared(Y, Y_hat):
	if len(Y.shape)==1:
		Y=Y.reshape(Y.shape[0],1)
	return 1- OLS(Y,Y_hat)/OLS(Y,Y.mean())


def OLS(Y, Y_hat):
	return np.sum((Y - Y_hat)**2)

def accuracy(y, y_hat):
	return np.mean(y == y_hat)


def precision(y, y_hat):
	return y.dot(y_hat)/y_hat.sum()


def recall(y, y_hat):
	return y.dot(y_hat)/y.sum()


def f1_score(y, y_hat):
	p = precision(y, y_hat)
	r = recall(y, y_hat)
	return 2*p*r/(p + r)


def confusion_matrix(Y, Y_hat):
	return np.matmul(Y.T, Y_hat)


def roc_auc(y, y_hat, p_hat, show_fig = True):
	pass

