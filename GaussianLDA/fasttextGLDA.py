# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(0)
from numpy import log, pi
from numba import jit
import random
from scipy.special import gammaln
import gensim
import math
from fasttextWishart import Wishart as Wishart
import cholesky as cholesky
import util as util
from corpus import *
corpus = Corpus()



def calculater(posterior, maxx, n_topics):                    
    for k in range(n_topics):
        expP = math.exp( posterior[k] - maxx )
        posterior[k] = expP
    return posterior




def return_max(logPosterior, maxx):
    if logPosterior > maxx:     
        maxx = logPosterior  
    return maxx



class GLDA(object):

    def __init__(self, n_topics, corpus, words2vec_ny, words2vec, vocab_ny, alpha = None):
        self.n_mk = None                          
        self.corpus = corpus                      # corpus.index2doc
        self.priors = None
        self.n_topics = n_topics                  
        self.vocab_ny = vocab_ny                        #set([])
        self.topic_params = defaultdict(dict)
        self.word_vec_size = words2vec.vector_size
        self.alpha = alpha #1/n_topics
        self.vocab2topic = {}
        self.words2vec = words2vec
        self.words2vec_ny = words2vec_ny
        self.solver = cholesky.Helper()
        self.AVGLD = util.CalculateComplexity()
        self.N_sum = 0
        self.logDeterminant = []
        self.z_mw = []
    
        assert isinstance(self.n_topics, int), 'n_topic should be an integer'

        
   




    def __updateTopicParameters(self, topic_id, word, removal):
        kappa_k, nu_k, df_k, scaleTdistr_k = self.__calculate_parameters(self.priors.kappa, self.priors.nu, self.n_k[topic_id], self.word_vec_size)
        
        newMean = self.topic_params[topic_id]["Topic Mean"]
        LT_old = self.topic_params[topic_id]["Lower Triangle"]
        words2vec_ny_w = self.words2vec_ny[word]
        
        if removal: 
            
            # downdate covariance matrix (sigma) or lower Triangular
            """ Sigma[n+1] =Sigma[n] - ((kappa_k+1)/kappa_k)*(X[n]-mu[n-1])(X[n]-mu[n-1])^T
            * therefore x = sqrt((kappa_k - 1)/kappa_k) * (X_{n} - \mu_{n})
            """
            centered = self.__downdate_centered(kappa_k, words2vec_ny_w, newMean)   # z
            LT_new = self.solver.chol_downdate(LT_old, centered)                    # Σ_k ← Σ_k - zz^T
            self.topic_params[topic_id]["Lower Triangle"] = LT_new
            
            # downdate topic parameter mu
            newMean = self.__downdate_mean(kappa_k, words2vec_ny_w, newMean)       
            self.topic_params[topic_id]["Topic Mean"] = newMean
            
        else: 
            # update topic parameter mu
            newMean = self.__update_mean(kappa_k, words2vec_ny_w, newMean)          
            self.topic_params[topic_id]["Topic Mean"] = newMean
            
            # update covariance matrix (sigma) or lower Triangular
            """
            Sigma[n+1] =Sigma[n] + ((kappa_k+1)/kappa_k)*(x[n+1]-mu[n+1])(x[n+1]-mu[n+1])^T
            """
            centered = self.__update_centered(kappa_k, words2vec_ny_w, newMean)     # z
            LT_new = self.solver.chol_update(LT_old, centered)                      # Σ_k ← Σ_k + zz^T
            self.topic_params[topic_id]["Lower Triangle"] =  LT_new
        
        logDet = self.__calculate_logDet(LT_new, scaleTdistr_k, self.word_vec_size)
        self.topic_params[topic_id]["Chol Det/Det(Sigma_k)"] = logDet       
        
        
    def __calculate_parameters(self, kappa_0, nu_0, n_k_k, dim):    
        kappa_k = kappa_0 + n_k_k
        nu_k = nu_0 + n_k_k
        df_k = nu_k - dim + 1
        scaleTdistr_k = (kappa_k + 1)/(kappa_k * df_k)
        return kappa_k, nu_k, df_k, scaleTdistr_k

    def __downdate_mean(self, kappa_k, words2vec_ny_w, newMean):
        newMean = newMean * (kappa_k + 1) 
        newMean = newMean - words2vec_ny_w                                    
        return newMean / kappa_k
    
    def __update_mean(self, kappa_k, words2vec_ny_w, newMean):
        newMean = newMean * (kappa_k - 1)  
        newMean = newMean + words2vec_ny_w                                    
        return newMean / kappa_k
    
    def __downdate_centered(self, kappa_k, words2vec_ny_w, newMean):
        centered =  words2vec_ny_w - newMean
        return centered * math.sqrt( (kappa_k + 1) / kappa_k)              
    
    def __update_centered(self, kappa_k, words2vec_ny_w, newMean):
        centered =  words2vec_ny_w - newMean
        return centered * math.sqrt(kappa_k / (kappa_k-1))                 
    
    def __downdate_counts(self, docID, wordCounter, old_k):   
        self.z_mw[docID][wordCounter] = -1 
        self.n_k[old_k] -= 1                                               
        self.n_mk[docID, old_k] -= 1    
        
    def __update_counts(self, docID, wordCounter, new_k):   
        self.z_mw[docID][wordCounter] = new_k
        self.n_k[new_k] += 1                                  
        self.n_mk[docID, new_k] += 1    
    
    def __calculate_logDet(self, Lower_Triangle_matrix, scale_factor, dim):
        logDet = np.sum(np.log(np.diag(Lower_Triangle_matrix)))
        return (logDet + dim * np.log(scale_factor) / 2.0 ) 
    
    def update_counter(self, new_k, old_k):
        if new_k != old_k:
            self.update_k_count += 1  
        
        
    def fit_initialize(self):    

        self.priors = Wishart(self.words2vec_ny, self.corpus, self.word_vec_size)
        scaleTdistr_0 = (self.priors.kappa + 1)/(self.priors.kappa * (self.priors.nu - self.word_vec_size + 1)) 
                                                                                               
        logDet = self.__calculate_logDet(self.priors.sigma, scaleTdistr_0, self.word_vec_size)
     
        for topic_id in range(self.n_topics):
            self.topic_params[topic_id]["Topic Mean"] = self.priors.mu
            self.topic_params[topic_id]["Lower Triangle"] = self.priors.sigma
            self.topic_params[topic_id]["Chol Det/Det(Sigma_k)"] = logDet    
        
        self.M = len(self.corpus.keys())    # number of docs
        self.N_sum = 0
        self.n_mk = np.zeros((self.M, self.n_topics), dtype ='intc')       
        self.n_k = np.zeros(self.n_topics, dtype ='intc')
        self.z_mw = []
        

        self.vocab2topic = {word: random.choice(range(self.n_topics)) for word in self.vocab_ny} # a topic assigned randomly to each word in vocab

        for docID, doc in self.corpus.items():              # docs in tokenized form
            self.N_sum += len(doc)                          # sum of total words in corpus
            z_m = []
            
            for word in doc:
                init_k = self.vocab2topic[word]             
                z_m.append(init_k)
                self.n_mk[docID, init_k] += 1               
                self.n_k[init_k] += 1                      
                self.__updateTopicParameters(init_k, word, False)
                
            self.z_mw.append(z_m)                          
            
           
        print('Number of tatal words is: ', self.N_sum, '  and number of documents in the corpus: ', self.M)
        print('n_k: ' ,self.n_k)
        avgLL = self.AVGLD.calculateAvgLogLikelihood(self.corpus, self.z_mw, self.words2vec_ny, self.topic_params, self.n_topics, self.word_vec_size, self.n_k, self.priors)
        print("Average LogDensity at the begining: ", avgLL)
        print('initialization completed')
        
        
 
        
        
        

    def fit(self, iterations=1):
        
        self.fit_initialize()
        
        
        for i in range(iterations):
            self.Collapsed_gibbs_sampling()
            print("{0}th iteration completed".format(i + 1))

        
        

        


        
    def Collapsed_gibbs_sampling(self):   
        self.update_k_count = 0
        
        for docID, doc in self.corpus.items():
            wordCounter = 0
            for word in doc:

                old_k = self.z_mw[docID][wordCounter]

                self.__downdate_counts(docID, wordCounter, old_k)                  #  update count-matrices
                self.__updateTopicParameters(old_k, word, True)                    #  updates params
                
                posterior = []
                maxx = -float('inf')    
                posteriorSum = 0
                
                for k in range(self.n_topics):
                                        
                    logprob = self.AVGLD.Generating_StudentTdistribution_pdf_in_logformat(self.words2vec_ny[word], k, self.topic_params, self.priors, self.word_vec_size, self.n_k)
                  
                    logPosterior = np.log(self.n_mk[docID ,old_k] + self.alpha) + logprob
                    posterior.append(logPosterior) 
                    
                    maxx = return_max(logPosterior, maxx)
                        
                posterior = calculater(posterior, maxx, self.n_topics)
                
                new_k = self.AVGLD.sampler(posterior) 

                self.update_counter(new_k, old_k)
                self.__update_counts(docID, wordCounter, new_k)                # update count-matrices       
                self.__updateTopicParameters(new_k, word, False)               # update params
                
                wordCounter += 1 
     
        print('n_k: ' ,self.n_k, '    sum of n_k: ', np.sum(self.n_k))
        avgLL = self.AVGLD.calculateAvgLogLikelihood(self.corpus, self.z_mw, self.words2vec_ny, self.topic_params, self.n_topics, self.word_vec_size, self.n_k, self.priors)
        print("Average LogDensity in this iteraion is : ", avgLL, '  The update rate of topics for this iteration is: ---> ', float(self.update_k_count/self.N_sum) )
        

                

    
    
                    
        

        
        
        
    
    
        

      
        
        
        