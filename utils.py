import numpy as np


def one_hot_encode(y):
	values=np.unique(y)
	return np.array([values==i for i in y])


def shuffle(*args):
	idx = np.random.permutation(len(args[0]))

	return [X[idx,:] for X in args]
