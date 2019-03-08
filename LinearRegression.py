
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#useful function
def SS(x,y):
    return x.T.dot(y)

class LinearRegression:
    def __init__(self):
        pass
    
    def fit(self, x,y, l1 = 0, l2 = 0, eta = 1e-3, epochs =1e4, show_curve=False):
        epochs=int(epochs)
        if len(x.shape)==1: x=x.reshape(x.shape[0],1)
        if len(y.shape)==1: y=y.reshape(y.shape[0],1)
        J=[]
        self.w=np.random.randn(x.shape[1])
        for t in range(epochs):
            y_hat = x.dot(self.w)
            J[t]=(SS(y,y_hat)+l1*np.sum(np.abs(self.w))+l2*SS(self.w,self.w))
            self.w -= eta*(x.T.dot(y_hat-y)+l1*np.sign(self.w) + l2*self.w)
        
        if show_curve:
            plt.figure()
            plt.plot(J)
            
        return self.w
    
    def predict(self, x):
        return x.dot(self.w)
    
    

