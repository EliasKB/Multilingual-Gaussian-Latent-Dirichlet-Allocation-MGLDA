# -*- coding: utf-8 -*-

import numpy as np
class Wishart(object):
    

    def __init__(self, word_vecs, docs, dim): 
        self.nu = None
        self.kappa = None
        self.mu = None
        self.set_params(word_vecs, docs, dim) 
        
    def set_params(self, word_vecs, docs, dim):
        counter = 0
        puck = np.zeros(( dim, ), dtype ='float64') 
        
        for idd , doc in docs.items():
            for word in doc:
                counter += 1
                puck += word_vecs[word]
                

        self.nu = dim + 1                      
        self.kappa = 0.1 
        self.sigma = np.identity( dim ) * 3    # changed this to identity matrix as in paper. No intuition here
        self.mu = puck/counter           