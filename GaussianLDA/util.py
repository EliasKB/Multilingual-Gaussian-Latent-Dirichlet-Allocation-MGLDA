# -*- coding: utf-8 -*-
import numpy as np
import random
np.random.seed(0)
from scipy import linalg
from numba import jit
import math
from numpy import log, pi
from scipy import linalg
from scipy.special import gammaln

class CalculateComplexity(object):
    

    def calculateAvgLogLikelihood(self, corpus, z_mw, words2vec_ny, topic_params, K, dim, n_k, priors):
        scaledLowerTriangles = []
        scaledlogDeterminants = []
        
        for k in range(K):
            scalar = priors.nu + n_k[k] - dim # + 1         df = nu-dim+1
            scaledLowerTriangle = topic_params[k]["Lower Triangle"]/math.sqrt(scalar)
            scaledLowerTriangles.append(scaledLowerTriangle)
            scaledlogDeterminants.append( np.sum(np.log(np.diag(scaledLowerTriangle))) )
            
            
        totalWordCounter = 0
        totalLogLL = 0
        docCounter = 0
        for docId, doc in corpus.items():
            wordCounter = 0
            for word in doc:
                x = words2vec_ny[word]                       
                topic_id = z_mw[docCounter][wordCounter]     
                b = x - topic_params[topic_id]["Topic Mean"] 
                X_muT_sigma_X_mu = self.__calculateX_muT_sigma_X_mu(scaledLowerTriangles[topic_id], b) # b^T sigma^-1 b=(L^-1 b)^T (L^-1 b)
                
                """ logDensity = log( (2pi)^(-k/2) * det(sigma)^(-1/2) * e^(-1/2 ((x-mu)^T sigma^-1  (x-mu)) ) ) """
                logDensity = self.__calculate_logDensity(X_muT_sigma_X_mu, dim, scaledlogDeterminants[topic_id] )
                
                totalLogLL -= logDensity
                wordCounter += 1
                totalWordCounter += 1
            docCounter += 1
        averageLogDensity = -totalLogLL / totalWordCounter
        
        return averageLogDensity
    
    def __calculate_logDensity(self, X_muT_sigma_X_mu, dim, scaledlogDeterminants_k ):
        return 0.5 * (X_muT_sigma_X_mu + dim * np.log(2 * np.pi)) + scaledlogDeterminants_k   
    
    
    def categorical_sampler(self, probs):  
        return(np.argmax(np.random.multinomial(1, pvals = probs)))
    
    

    def Generating_StudentTdistribution_pdf_in_logformat(self, words2vec_ny_word, topic_id, topic_params, priors, P, n_k):
        kappa_k, nu_k, df_k, scaleTdistribution_k = self.__calculate_parameters(priors.kappa, priors.nu, n_k[topic_id], P)
        log_cov_det_k = topic_params[topic_id]["Chol Det/Det(Sigma_k)"]                             #2*Sum_i=1^M(log(L_i,i))
        
        b = words2vec_ny_word - topic_params[topic_id]["Topic Mean"]          
        scaledLowerTriangle_k = topic_params[topic_id]["Lower Triangle"] * scaleTdistribution_k
        X_muT_sigma_X_mu = self.__calculateX_muT_sigma_X_mu(scaledLowerTriangle_k, b)      
        
        return self.__calculate_log_probs(df_k, P, log_cov_det_k, X_muT_sigma_X_mu)

    
    def __calculate_parameters(self, kappa_0, nu_0, n_k_k, dim):    
        kappa_k = kappa_0 + n_k_k
        nu_k = nu_0 + n_k_k
        df_k = nu_k - dim + 1
        scaleTdistr_k = math.sqrt((kappa_k + 1)/(kappa_k * df_k))
        return kappa_k, nu_k, df_k, scaleTdistr_k  
    
    
    def __calculateX_muT_sigma_X_mu(self, scaledLowerTriangle_k, b):
        L = linalg.cholesky(scaledLowerTriangle_k)                                   # Cholesky decomposition     A = LL*
        X = linalg.cho_solve((L, True), b, overwrite_b=True, check_finite=True)      # solve x = A^-1 * b in A x = b 

        return X.T.dot(X) 
   

    
    def __calculate_log_probs(self, df, P, log_cov_det_k, XTX):   
        # return log_prob
        return ( gammaln((df + P) / 2.) - (  gammaln(df/2.) + P/2. * ( np.log(df) + np.log(np.pi) ) + log_cov_det_k + ((df+P)/2.) * np.log(1. + XTX  / df)) )
        
    
    
 
    
    
    
    
    def binSearch(self, cumProb, key, start, end):
        if start > end:
            return start
        mid = int((start + end)/2)
        
        if key == cumProb[mid]:
            return mid+1
        if key < cumProb[mid]:
            return self.binSearch(cumProb, key, start, mid-1)        
        if key > cumProb[mid]:
            return self.binSearch(cumProb, key, mid+1, end) 
        return -1
    
    def sampler(self, arrP):
        arrP2 = arrP
        cumP = []
        sum_probs = 0
        for prob in arrP:
            sum_probs = sum_probs + prob
            cumP.append(sum_probs)
        if sum_probs != 1:
            for i in range(len(arrP)):
                arrP2[i] = arrP2[i]/sum_probs
                cumP[i] = cumP[i]/sum_probs
        
        r = np.random.uniform(0,1,1)
        return ( self.binSearch(cumP, r, 0, int(len(cumP)-1)) )
        
    
                


                    
            
            
            
            

    
    
    
    
    
    
    


