# -*- coding: utf-8 -*-

# Please dont use this file for filtering of corpus of non-latin languages

""" Run the commented code below in case u have not downloaded the nltk before """
#nltk.download('stopwords')            # need to be downloaded if u have not it already
#nltk.download('wordnet')

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

class filtering(object):
    def __init__(self, stemmer, stop_words): 
        self.stemmer = stemmer
        self.stop_words = stop_words

    def lemmatize_stemming(self, word):
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(word, pos='s')) # pos='v'
    
    def preprocessing(self, documents):
        bad_chars = [';', ':','.',',','<','>','``','`´','´´','`','´','{','}','[',']','(',')','"','^',"'"," 's","*",'@','\\','$','~',
                     '#','&', '!','?','-','+','0','1','2','3','4','5','6','7','8','9','=', '%','|','/','_',"'"]

        new_documens = [];
        for doc in documents:
            for char in bad_chars :
                doc = doc.replace(char, '')    	# replace bad chars by space in storytel data and empty space in Amazon data
            txt = doc.split()
            txt = map(lambda x: x.lower(), txt)                         # Lowercasing each word
            txt = filter(lambda x: x not in self.stop_words, txt)        # Removing stop words as "can", "should", "of", "in" 
            txt = filter(lambda x: len(x) > 1, txt)                      # removing super short words and single digit/letter
            #txt = [self.lemmatize_stemming(word) for word in txt]       # transforming each word ro original form, plural to sing
            txt = ' '.join(txt)
            new_documens.append(txt)
            
        print('filteringsprocess is completed')
        return(new_documens)
