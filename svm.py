# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 18:07:00 2022

@author: ssingh17
"""

import numpy as np


class SVM:

    
    def __init__(self, learning_rate = 0.01, lang_mult = 0.01, n_iters = 100):
        self.learning_rate = learning_rate
        self.lang_mult = lang_mult
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
        
    def fit(self, X, y):
        n_samples, n_features = X.shape        
        y = np.where(y <= 0, -1, 1)        
        self.w = np.zeros(n_features)
        self.b = 0
                
        for j in range(self.n_iters):            
            for i in range(n_samples):                
                self.w = self.w - self.learning_rate*(self.w - self.lang_mult*np.dot(X[i], y[i]))
                self.b = self.b - self.learning_rate*self.lang_mult*y[i]

                
    def predict(self, X):
        value = np.dot(X, self.w) + self.b
        return np.sign(value)



