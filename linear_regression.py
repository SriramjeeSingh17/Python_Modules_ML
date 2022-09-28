# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 11:43:54 2022

@author: ssingh17
"""


import numpy as np


class LR:
    
    def __init__(self):
        self.beta = None
        
          
    def fit(self, X, y):
        m = X.shape[0]
        X = np.append(X, np.ones(m).reshape(m, 1), axis = 1)
        X, y = np.matrix(X), np.matrix(y)
        self.beta = np.linalg.pinv((X.T)*X)*((X.T)*y)
        
        
    def predict(self, X):
        m = X.shape[0]
        X = np.append(X, np.ones(m).reshape(m, 1), axis = 1)
        pred = np.dot(X, self.beta)
        return np.array(pred)
        
        
        

