import numpy as np
import matplotlib.pyplot as plt

from objectives import *
from metrics import *
from utils import *


def sigmoid(H):
	return 1/(1 + np.exp(-H))


def softmax(H):
	eH = np.exp(H)
	return eH/eH.sum(axis = 1, keepdims = True)


def ReLU(H):
	return H*(H > 0)


def derivative(a):
	if a is sigmoid:
		return lambda x: x*(1 - x)
	elif a is np.tanh:
		return lambda x: 1 - x*x
	elif a is ReLU:
		return lambda x: x > 0
	else:
		raise exception("No known activation provided.")


class ANN():
	def __init__(self, hidden_layer_sizes, hidden_activations = None):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.hidden_activations = hidden_activations
		self.L = len(hidden_layer_sizes) + 1

	def forward(self, X, dropout=0):
		self.Z = {0: X}
		N=X.shape[0]
		for l in sorted(self.a.keys()):
			if dropout>0: self.Z[l-1] *= (np.random.rand(N,self.Z[l-1].shape[1])>dropout)/(1-dropout)
			self.Z[l] = self.a[l](np.matmul(self.Z[l - 1],self.W[l]) + self.b[l])


	def fit(self, X, y, eta = 1e-3, Logistic =True, lambda2 = 0, lambda1 = 0, epochs = 10, show_curve = False, 
			batch_sz = 50, mu = .9, gamma = .9, adam =False, epsilon = 1e-10, noise= 1e-10, dropout=.01):

		if (len(X.shape)==1): 
			X =X.reshape((X.shape[0],1))

		if (len(y.shape)==1): 
			y = y.reshape((y.shape[0],1))

		self.Xcenter=X.mean()
		self.Xscale=max(1,X.max()-X.mean(), X.mean()-X.min())
		X=(X-self.Xcenter)/self.Xscale

		N, D = X.shape
		if Logistic:
			K = len(set(y))
			Y = one_hot_encode(y)
		else:
			K = y.shape[1]
			Y = y.copy()
		

		self.layer_sizes = [D] + self.hidden_layer_sizes + [K]

		self.W = {l + 1: np.random.randn(M[0],M[1]) for l, M in enumerate(zip(self.layer_sizes, self.layer_sizes[1:]))}
		self.b = {l + 1: np.random.randn(M) for l, M in enumerate(self.layer_sizes[1:])}


		if self.hidden_activations is None:
			self.a = {l+1: ReLU for l in range(self.L - 1)}
		elif type(self.hidden_activations) != type([]):
			self.a = {l+1: self.hidden_activations for l in range(self.L - 1)}
		elif len(self.hidden_activations) == 1:
			self.a = {l+1: self.hidden_activations[0] for l in range(self.L - 1)}
		else:
			self.a = {l+1: act for l, act in enumerate(self.hidden_activations)}

		self.d_a = {l:derivative(self.a[l]) for l in self.a.keys()}
		if Logistic:
			self.a[self.L] = softmax
		else:
			self.a[self.L] = lambda h: h

		J = []

		utility = self.get_utility_function(Y, lambda1, lambda2, Logistic)
		d_W, d_b = self.get_update_function(eta, mu, gamma, epsilon, adam, lambda1, lambda2)
		

		if batch_sz > N: batch_sz = N

		for epoch in range(int(epochs)):
			X, Y, y = shuffle(X.copy(), Y.copy(), y.copy())
			for idx in range(N//batch_sz):
				x_b = X[idx*batch_sz:(idx +1)* batch_sz,:]
				y_b = Y[idx*batch_sz:(idx +1)* batch_sz,:]

				self.forward(x_b + np.random.randn(batch_sz,D) * noise, dropout=dropout)

				dH = self.Z[self.L] - y_b

				for l in sorted(self.W.keys(), reverse = True):
					dW = np.matmul(self.Z[l - 1].T,dH)
					self.W[l] -= d_W( l, dW)
					self.b[l] -= d_b( l, dH.sum(axis = 0) )

					if l > 1:
						dZ = np.matmul(dH, self.W[l].T)
						dH = dZ*self.d_a[l-1](self.Z[l - 1])
			if show_curve:		
				self.forward(X)
				J.append(utility(Y))
		if show_curve:
			plt.plot(J[20:])
			plt.title("Training Curve")
			plt.xlabel("epochs")
			plt.ylabel("J")
			plt.show()

	def predict(self, X):
		X=(X-self.Xcenter)/self.Xscale
		self.forward(X)
		return self.Z[self.L]

	def get_utility_function(self, Y, lambda1, lambda2, Logistic):
		if Logistic:
			cost_funct = lambda Y: cross_entropy(Y,self.Z[self.L])
		else:
			cost_funct = lambda Y: OLS(Y,self.Z[self.L])
		if lambda1 > 0: util_function = lambda Y: cost_funct(Y) + (lambda2/2)*sum(np.sum(W*W) for W in self.W.values())
		elif lambda2 >0: util_function = lambda Y: cost_funct(Y) + (lambda1)*sum(np.sum(np.abs(W)) for W in self.W.values())
		else: util_function = lambda Y: cost_funct(Y) + (lambda1)*sum(np.sum(np.abs(W)) for W in self.W.values()) + (lambda2/2)*sum(np.sum(W*W) for W in self.W.values())
		return util_function

	def get_update_function(self, eta, mu, gamma, epsilon, adam, lambda1, lambda2):
		#derivative with and without lambda regularization
		if (lambda1==0) and (lambda2 == 0):
			deltaW = lambda dW: dW
		elif lambda1 == 0: 
			deltaW = lambda dW: dW + (lambda2/2)*sum(np.sum(W) for W in self.W.values())
		elif lambda2 == 0: 
			deltaW = lambda dW: dW + (lambda1)*sum(np.sum(np.sign(W)) for W in self.W.values())
		else:
			deltaW = lambda dW: dW + (lambda1)*sum(np.sum(np.sign(W)) for W in self.W.values()) + (lambda2/2)*sum(np.sum(W) for W in self.W.values())
		
		#if no momemtum 
		if (mu == 0) and (gamma ==0):
			d_W = lambda l, dW: eta * deltaW(dW)
			d_b = lambda l, db: eta * db
			return d_W, d_b
		
		#if RMSProp or ADAM
		if gamma == 0:
			epsilon = 0
			g_W = lambda l, dW: 1
			g_b = lambda l, db: 1
		else:
			self.g_W = {i + 1:0 for i in range(self.L)}
			self.g_b = {i + 1:0 for i in range(self.L)}

			def g_W(l, dW):
				self.g_W[l] = gamma * self.g_W[l] + (1 - gamma) * deltaW(dW)**2
				return self.g_W[l]

			def g_b(l, db):
				self.g_b[l] = gamma * self.g_b[l] + (1 - gamma) * db**2
				return self.g_b[l]

		if mu == 0:
			d_W =  lambda l, dW: eta * deltaW(dW) / np.sqrt(g_W(l,dW) + epsilon)
			d_b =  lambda l, db: eta * db / np.sqrt(g_b(l,db) + epsilon)
			return d_W, d_b
		
		self.vel_W = {i + 1:0 for i in range(self.L)}
		self.vel_b = {i + 1:0 for i in range(self.L)}

		if adam:
			self.t=1
			def d_W( l, dW):
				self.vel_W[l] = mu * self.vel_W[l] + (1 - mu) * deltaW(dW)
				return eta * self.vel_W[l] / (1 - mu**self.t) / np.sqrt(g_W(l,dW) / (1 - gamma**self.t)+ epsilon)

			def d_b( l, db):
				self.vel_b[l] = mu * self.vel_b[l] + (1 - mu) * db
				return eta * self.vel_b[l] / (1 - mu**self.t) / np.sqrt(g_b(l,db) / (1 - gamma**self.t)+ epsilon)
			return d_W, d_b
		else:
			def d_W( l, dW):
				self.vel_W[l] = mu * self.vel_W[l] + deltaW(dW)
				return eta * self.vel_W[l] / np.sqrt(g_W(l,dW) + epsilon)

			def d_b( l, db):
				self.vel_b[l] = mu * self.vel_b[l] + db
				return eta * self.vel_b[l] / np.sqrt(g_b(l,db) + epsilon)
			return d_W, d_b