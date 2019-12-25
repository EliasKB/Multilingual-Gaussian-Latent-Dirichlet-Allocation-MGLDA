# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
np.random.seed(0)
from scipy import linalg
from numba import jit
import math


# Pseudo code can be found in Wikipedia https://en.wikipedia.org/wiki/Cholesky_decomposition
class Helper(object):
    @jit(cache=True)
    def chol_update(self, L, X):         # see paper (das et al. 2015) page 798,     A' = A + x*x^T 
        assert L.shape[1] == X.shape[0], "cholesky lower triangle matrix dim != word vec dim"
        L_c = np.copy(L)
        for k in range(X.shape[0]):
            r = np.sqrt(L_c[k, k] ** 2 + X[k] ** 2)
            c = r / L_c[k, k]
            s = X[k] / L_c[k, k]
            L_c[k, k] = r
            for i in range(k+1, X.shape[0]):
                L_c[i, k] = (L_c[i, k] + s * X[i]) / c    
                X[i] = c * X[i] - s * L_c[i, k]
        return L_c
    
    
           
    @jit(cache=True)
    def chol_downdate(self, L, X):       # see paper page 798,     A' = A - x*x^T 3
        assert L.shape[1] == X.shape[0], "cholesky lower triangle matrix dim != word vec dim"
        L_c = np.copy(L) 
        for k in range(X.shape[0]):
            r = np.sqrt(L_c[k, k]**2 - X[k]**2)               
                
            c = r / L_c[k, k]
            s = X[k] / L_c[k, k]
            L_c[k, k] = r
            for i in range(k+1, X.shape[0]):
                L_c[i, k] = (L_c[i, k] - (s * X[i])) / c     
                X[i] = c * X[i] - s * L_c[i, k]
        return L_c
            
            

    
    
    
    
    
    
    


