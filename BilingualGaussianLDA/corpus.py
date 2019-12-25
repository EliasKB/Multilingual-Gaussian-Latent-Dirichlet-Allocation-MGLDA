# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim import utils
np.random.seed(0)

class Corpus(object):

    def __init__(self):
        
        # OBS: Some of theses tools will not be used in GLDA
        
        self.id2word = {}
        self.word2id = {}
        self.index2doc = {}
        self.index2doc_test = {}
        self.docs_idform = []
        self.docs = []
        self.vocab = set([])
        self.vocab_ny = set([])
        self.temp_corpus = defaultdict(dict)
        self.words2vec_ny = {}
        self.valid_split = 0
        self.V = 0
        self.M = 0
        
        assert self.valid_split >= 0 and self.valid_split < 1, 'valid_split should be in interval: [0, 1)'

    
    
    """ Each document is considered as a line and words in a documents are separated by whitespace. """
    
    def load_text(self, filepath,  valid_split = 0): 
        self.valid_split = valid_split
        
        input_file = open(filepath, 'r')
        
        temp_docs = []
        for line in input_file:
            temp_docs.append(line.strip().split(' '))
        M = int((1 - self.valid_split) *  len(temp_docs)) 
        temp_docs = temp_docs[:M]
       
        
        for index_doc, doc in enumerate(temp_docs):
            
            """  making docs , index2doc and temp_corpus  """
            self.docs.append(doc)
            self.index2doc[index_doc] = doc                       # docs in tokenized form
            
            doc_id = np.empty(len(doc), dtype='intc')
            
            
            """  making vocab , word2id , id2word  and  doc_idform  """
            for index, word in enumerate(doc):          
                self.vocab.add(word)                              
                if word not in self.word2id:                      # if unseen
                    current_id = len(self.word2id)    
                    self.word2id[word] = current_id               # word2id is a dictionary of words and ids
                    self.id2word[current_id] = word               # id2word is a dictionay of ids and words 

                    doc_id[index] = current_id 
                else:                                             # if seen
                    doc_id[index] = self.word2id[word]            # doc_id is a array of id (unique)
            self.docs_idform.append(doc_id)
            

        self.V = len(self.word2id)                                # number of unique words in whole of corpus
        self.M = len(self.docs_idform)                            # numer of docs in corpus
        print("Done processing corpus with {} documents".format(len(self.docs_idform)))

        input_file.close() 
    
    
    
    def process_wordvectors_ny(self, modelvw):               
        """ making words2vec_ny of dictioanry format in pretrained case when we train embedding by fasttext """
        self.words2vec_ny = {}
        for word in self.vocab:                                    
            self.words2vec_ny[word] = modelvw[word]
        
        
      
    def process_wordvectors(self, filepath = None):        
        """ making words2vec_ny , vocab_ny: 
            Changes the format of embedding from vec or txt into word2vec-format
            Makes a new vocabualry of the corpus only from the words that r presented in the pre-trained embedding, shrinked vocab
        """

        self.words2vec = filepath   
        useable_vocab = 0
        unusable_vocab = 0

        self.words2vec_ny = {}
        for word in self.vocab:                                    
            try:
                self.words2vec_ny[word] = self.words2vec[word]      
                useable_vocab += 1
                self.vocab_ny.add(word)                             
            except KeyError:
                unusable_vocab += 1
                
                
        print('len(words2vec_ny): ' , len(self.words2vec_ny)) 
        for index_doc, doc in self.index2doc.items(): 
            self.index2doc[index_doc] = [word for word in doc if word in self.vocab_ny]
        print('vocab_old: ',len(self.vocab), '      vocab_ny: ',len(self.vocab_ny))
        print('-----------')
        
        self.word_vec_size = self.words2vec.vector_size             # 50
        print('self.words2vec.vector_size: ', self.words2vec.vector_size, '   len(self.words2vec_ny) : ', len(self.words2vec_ny))
       

        print("There are {0} words that could be converted to word vectors in your corpus \n" \
                  "There are {1} words that could NOT be converted to word vectors".format(useable_vocab, unusable_vocab))
        
        
        
        
  
    def convert_dictionary_to_words2vec(self, fname):
        """ we need dictioany of word2vec format to be able to use words2vec.most_similar() function for printing n_top words of resp. topic """
        vocab=self.words2vec_ny
        vectors = self.words2vec_ny.values()
        vector_size=self.word_vec_size
        total_vec=len(self.words2vec_ny)
    
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
            for word, row in vocab.items():
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))
        words2vec2 = KeyedVectors.load_word2vec_format(fname)
        return(words2vec2)
        
     
    
    
    
 

""" Example
input = ["apple orange mango melon", "dog cat bird rat",]

outputs: 
1. vocab = {'apple', 'bird', 'cat', 'dog', 'mango', 'melon', 'orange', 'rat'}       
        
2. id2word = {0: 'apple', 1: 'orange', 2: 'mango', 3: 'melon', 4: 'dog', 5: 'cat', 6: 'bird', 7: 'rat'}

3. word2id = {'apple': 0, 'orange': 1, 'mango': 2, 'melon': 3, 'dog': 4, 'cat': 5, 'bird': 6, 'rat': 7}

4. docs_idform = [array([0, 1, 2, 3], dtype=int32), array([4, 5, 6, 7], dtype=int32)]

5. docs = [['apple', 'orange', 'mango', 'melon'], ['dog', 'cat', 'bird', 'rat']] 

6. temp_corpus = defaultdict(dict,     {  0: {'words': ['apple', 'orange', 'mango', 'melon'], 'topics': array([5, 3, 2, 0])},
                                       1: {'words': ['dog', 'cat', 'bird', 'rat'], 'topics': array([5, 1, 0, 4])}})

7. vocab = {'apple', 'bird', 'cat', 'dog', 'mango', 'melon', 'orange', 'rat'}


uses for the next steps


8. index2doc = {0: ['apple', 'orange', 'mango', 'melon'], 1: ['dog', 'cat', 'bird', 'rat']}

9. vocab_ny =  {'apple', 'bird', 'cat', 'dog', 'mango', 'melon', 'orange', 'rat'} 

nackdel: unseen word
fordel: sparar minne

10. corpus.words2vec_ny = 
{'orange': array([-0.42783 ,  0.43089 , -0.50351 ,  0.5776  ,  0.097786,  0.2608  ,
        -0.68767 , -0.31936 , -0.25337 , -0.37255 , -0.045907, -0.53688 ,
         0.97511 , -0.44595 , -0.50414 , -0.086751, -1.0645  ,  0.36625 ,
        -0.52428 , -1.3413  , -0.2391  , -0.58808 ,  0.56378 , -0.062501,
        -1.7429  , -0.88077 , -0.27933 ,  1.4705  ,  0.50436 , -0.69174 ,
         2.0018  ,  0.26663 , -0.85679 , -0.18893 , -0.021125, -0.055118,
        -0.50337 , -0.67157 ,  0.55502 , -0.8009  ,  0.10695 ,  0.1459  ,
        -0.55588 , -0.64971 ,  0.22046 ,  0.67415 , -0.45119 , -1.1462  ,
         0.16348 , -0.62946 ], dtype=float32),
 'melon': array([ 0.18441  , -0.53967  , -0.88737  ,  0.68819  ,  0.28368  ,
        -0.018605 ,  0.26348  ,  0.25444  ,  0.086219 ,  0.89426  ,
         0.10597  ,  0.1046   ,  0.70401  ,  0.47789  ,  0.2736   ,
        -0.54892  ,  0.68631  ,  0.72113  , -0.25797  , -0.73963  ,
        -0.2118   , -0.61244  ,  1.5459   ,  0.19133  , -0.12898  ,
         1.042    , -0.48443  ,  0.68686  , -0.10321  , -0.56111  ,
         0.51709  , -0.47882  , -0.48083  ,  1.4443   ,  0.27575  ,
         0.35497  , -1.2676   ,  0.52617  ,  1.0308   , -0.64129  ,
        -0.26462  , -0.10706  , -0.68828  , -0.44401  ,  0.93317  ,
        -0.28855  ,  1.0842   , -0.37766  ,  0.0081279, -0.49466  ],
       dtype=float32),
'cat': array([ 0.45281 , -0.50108 , -0.53714 , -0.015697,  0.22191 ,  0.54602 ,
        -0.67301 , -0.6891  ,  0.63493 , -0.19726 ,  0.33685 ,  0.7735  ,
         0.90094 ,  0.38488 ,  0.38367 ,  0.2657  , -0.08057 ,  0.61089 ,
        -1.2894  , -0.22313 , -0.61578 ,  0.21697 ,  0.35614 ,  0.44499 ,
         0.60885 , -1.1633  , -1.1579  ,  0.36118 ,  0.10466 , -0.78325 ,
         1.4352  ,  0.18629 , -0.26112 ,  0.83275 , -0.23123 ,  0.32481 ,
         0.14485 , -0.44552 ,  0.33497 , -0.95946 , -0.097479,  0.48138 ,
        -0.43352 ,  0.69455 ,  0.91043 , -0.28173 ,  0.41637 , -1.2609  ,
         0.71278 ,  0.23782 ], dtype=float32),
 'bird': array([ 0.78675 ,  0.079368, -0.76597 ,  0.1931  ,  0.55014 ,  0.26493 ,
        -0.75841 , -0.8818  ,  1.6468  , -0.54381 ,  0.33519 ,  0.44399 ,
         1.089   ,  0.27044 ,  0.74471 ,  0.2487  ,  0.2491  , -0.28966 ,
        -1.4556  ,  0.35605 , -1.1725  , -0.49858 ,  0.35345 , -0.1418  ,
         0.71734 , -1.1416  , -0.038701,  0.27515 , -0.017704, -0.44013 ,
         1.9597  , -0.064666,  0.47177 , -0.03    , -0.31617 ,  0.26984 ,
         0.56195 , -0.27882 , -0.36358 , -0.21923 , -0.75046 ,  0.31817 ,
         0.29354 ,  0.25109 ,  1.6111  ,  0.7134  , -0.15243 , -0.25362 ,
         0.26419 ,  0.15875 ], dtype=float32),
'rat': array([ 0.80165  , -0.36917  , -0.38166  , -0.090699 , -0.48203  ,
         0.44434  , -0.093362 , -0.91821  ,  0.56248  , -0.1119   ,
         0.18622  ,  0.75844  ,  0.57118  ,  0.88046  ,  0.039176 ,
         0.14085  ,  0.86972  ,  0.9068   , -1.0038   ,  0.39386  ,
        -0.97257  , -0.80421  ,  1.1652   ,  0.437    ,  0.27361  ,
        -1.1458   , -0.78352  ,  0.76388  , -0.27542  , -0.67853  ,
         0.75349  , -0.072083 , -0.32882  ,  0.34472  , -0.52561  ,
         0.56328  ,  0.0089467, -0.12906  ,  1.1168   ,  0.072277 ,
        -0.49204  ,  0.7962   , -0.65754  ,  0.61735  ,  1.2019   ,
        -0.46379  ,  0.54618  , -1.0997   ,  0.85173  ,  0.04412  ],
       dtype=float32),
 'apple': array([ 0.52042 , -0.8314  ,  0.49961 ,  1.2893  ,  0.1151  ,  0.057521,
        -1.3753  , -0.97313 ,  0.18346 ,  0.47672 , -0.15112 ,  0.35532 ,
         0.25912 , -0.77857 ,  0.52181 ,  0.47695 , -1.4251  ,  0.858   ,
         0.59821 , -1.0903  ,  0.33574 , -0.60891 ,  0.41742 ,  0.21569 ,
        -0.07417 , -0.5822  , -0.4502  ,  0.17253 ,  0.16448 , -0.38413 ,
         2.3283  , -0.66682 , -0.58181 ,  0.74389 ,  0.095015, -0.47865 ,
        -0.84591 ,  0.38704 ,  0.23693 , -1.5523  ,  0.64802 , -0.16521 ,
        -1.4719  , -0.16224 ,  0.79857 ,  0.97391 ,  0.40027 , -0.21912 ,
        -0.30938 ,  0.26581 ], dtype=float32),
'mango': array([ 0.26381 , -0.31832 , -1.0953  ,  1.3305  ,  0.24761 ,  0.045313,
        -0.39509 , -0.52107 , -0.016796,  0.33175 , -0.53252 ,  0.43263 ,
         1.2306  , -0.36963 ,  0.15989 , -0.433   , -0.29768 ,  0.768   ,
         0.71255 , -0.85675 , -0.076953, -1.0284  ,  0.9337  ,  0.24969 ,
        -0.13985 ,  1.0316  , -0.15809 ,  0.80512 ,  0.50535 , -0.50557 ,
         1.1237  , -0.45083 , -0.27552 ,  1.3537  ,  0.3553  ,  0.39403 ,
        -1.1213  ,  0.027925,  0.57582 , -0.63611 , -0.53506 , -0.080186,
        -0.78026 , -1.1595  ,  1.0318  ,  0.94337 ,  0.026387, -0.96839 ,
         0.54497 , -0.16479 ], dtype=float32),
 'dog': array([ 0.11008  , -0.38781  , -0.57615  , -0.27714  ,  0.70521  ,
         0.53994  , -1.0786   , -0.40146  ,  1.1504   , -0.5678   ,
         0.0038977,  0.52878  ,  0.64561  ,  0.47262  ,  0.48549  ,
        -0.18407  ,  0.1801   ,  0.91397  , -1.1979   , -0.5778   ,
        -0.37985  ,  0.33606  ,  0.772    ,  0.75555  ,  0.45506  ,
        -1.7671   , -1.0503   ,  0.42566  ,  0.41893  , -0.68327  ,
         1.5673   ,  0.27685  , -0.61708  ,  0.64638  , -0.076996 ,
         0.37118  ,  0.1308   , -0.45137  ,  0.25398  , -0.74392  ,
        -0.086199 ,  0.24068  , -0.64819  ,  0.83549  ,  1.2502   ,
        -0.51379  ,  0.04224  , -0.88118  ,  0.7158   ,  0.38519  ],
       dtype=float32)}
"""