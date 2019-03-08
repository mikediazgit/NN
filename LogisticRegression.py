
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#useful functions
def softmax(H):
    eH=np.exp(H)
    return eH / eH.dot(np.ones((H.shape[1],eH.shape[1])))

def cross_entropy(Y,P):
    return -np.sum(Y*np.log(P))

def accuracy(Y,P):
    return np.mean(Y.argmax(axis=1)==P.argmax(axis=1))

#useful function
def SS(x,y):
    return x.T.dot(y)

class LogisticRegression:
    def __init__(self):
        pass
    
    def fit(self, x,y, l1 = 0, l2 = 0, eta = 1e-3, epochs =1e4):
        epochs=int(epochs)
        if len(x.shape)==1: x=x.reshape(x.shape[0],1)
        if len(y.shape)==1: y=y.reshape(y.shape[0],1)
        J=[]
        self.w=np.random.randn(x.shape[1], y.shape[1])
        for e in range(epochs):
            P = softmax(x.dot(self.w))
            J.append(cross_entropy(y,P))
            self.w -= eta * x.T.dot(P - y)
            
        return self.w
    
    def predict(self, x):
        return softmax(x.dot(self.w))
    
    

