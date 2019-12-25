# -*- coding: utf-8 -*-
import timeit
import os
import numpy as np
from filtering import *
from corpus import Corpus
from fasttextGLDA import *
from nltk import corpus as nltk_corpus
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models import FastText


def read_documents(reviews):
    docs = []; labels = []; 
    with open(reviews, encoding = 'utf-8') as file: 
        for line in file:
            label, _, _, doc = line.strip().split( maxsplit = 3 )
            docs.append(doc)
            labels.append(label)
    return docs, labels


def read_documents_novel(urpath, fileType_or_fileName):
    docs = []
    for subdir, dirs, files in os.walk( urpath + 'data/other_data/'):
        for i in range(len(files)):
            filepath = subdir + os.sep + files[i]
            if filepath.endswith( fileType_or_fileName ):
                with open(filepath, encoding = 'utf-8') as f:
                    contents = f.read()
                    docs.append(contents)
    return docs



""" Define ur path there u have the folders of the data """
urpath = 'C:/Users/elias/OneDrive/Skrivbord/github/thesis/'



language_of_corpus = 'english' # alternative swedish


""" OBS: There are stemmer and stop_words for European languagres but not for Persian, Arabic etc. """
stemmer = SnowballStemmer(language_of_corpus)
stop_words = set(nltk_corpus.stopwords.words(fileids = language_of_corpus))
#Alternative: stop_words = set(stopwords.words('english')) or = stopwords.words('english') 




""" in case the corpus is Amazon reviews. For using other files such as novels etc. set it to False """
Amazon_reviews = True


""" Whether u want to use pre-trained embedding or create/train ur own embedding on the same corpus u later will train GLDA on. """
pre_trained_embedding = False




# u can change the hyper-parameters to the ones in the thesis in order to get similar results.
""" rate_usageOfData is between [0, 1]. 1.0 to train GLDA on entire corpus. 
OBS: in trained embedding case the embeddding will be build on entire corpus in anyway """
rate_usageOfData_Amazon = 0.02
rate_usageOfData_novels = 1.0
learning_rate = 0.05
dim_trained_embedding = 50
n_topic = 20
n_gibbs_iteration = 2



if __name__ == "__main__":

    """  Reading the data """
    if Amazon_reviews:
        docs, labels = read_documents(urpath + 'data/Amazon_data/amazon-English-EntireData.txt')
        """ Alternative: smaller corpus: 
            data/Amazon_data/amazon-English-first10%.txt
            data/Amazon_data/amazon-English-second10%.txt
            data/Amazon_data/amazon-Swedish-first10%.txt
            data/Amazon_data/amazon-Swedish-second10%.txt
            amazon-Persian-first10%.txt  # Use another pre-processings method for Persian corpus
        """
        
    else:
        docs = read_documents_novel(urpath, ".txt")  # .csv, .pdf,... or type the name a file



    """  Pre-processing the data such as filtering/cleaning etc. """
    f = filtering(stemmer, stop_words)
    filtered_docs = f.preprocessing(docs)

    with open(urpath + "temporary_files/docs_filtered.txt", "w") as output:
        for doc in filtered_docs:
            output.write('%s\n' % doc)
        
    corpus = Corpus()
    

    if Amazon_reviews:
        corpus.load_text(urpath + "temporary_files/docs_filtered.txt", valid_split = 1 - rate_usageOfData_Amazon )
    else:
        corpus.load_text(urpath + "temporary_files/docs_filtered.txt", valid_split = 1 - rate_usageOfData_novels )
    

    if pre_trained_embedding:
        
        """ These files cantain pre-trained embeddings and should be downloaded manually from the given links. """
        gensim_file = urpath + 'data/wiki.en.simple.vec'         # English   https://fasttext.cc/docs/en/pretrained-vectors.html
        
        
        """ Alternativt: other pre-trained embeddings  
            gensim_file = urpath + 'data/wiki-news-300d-1M.vec'     # English   https://fasttext.cc/docs/en/english-vectors.html
            gensim_file = urpath + 'data/wiki.en.vec'               # English   https://fasttext.cc/docs/en/pretrained-vectors.html
            gensim_file = urpath + 'data/wiki.sv.vec'               # Swedish   https://fasttext.cc/docs/en/pretrained-vectors.html
            gensim_file = urpath + 'data/wiki.fa.vec'               # Persian   https://fasttext.cc/docs/en/pretrained-vectors.html
        """

 

        corpus.process_wordvectors(filepath = gensim_file)
        shrinkedEmbedding_OfWord2vecFormat = corpus.convert_dictionary_to_words2vec(fname = urpath + 'temporary_files/train.txt')
        model = GLDA(n_topics = n_topic, corpus=corpus.index2doc, words2vec_ny=corpus.words2vec_ny, words2vec=corpus.words2vec, \
             vocab_ny = corpus.vocab_ny, alpha = learning_rate)

        
    else:
        """ train and create word enbedding by Facebook's fasttext from the current data """
        modelvw = FastText(corpus.docs, size = dim_trained_embedding , window=3, min_count=1,workers=5, alpha = 0.1,\
                   iter = 10, sg = 1, word_ngrams=1)
        model = GLDA(n_topics = n_topic, corpus=corpus.index2doc, words2vec_ny=modelvw, words2vec=modelvw ,vocab_ny = corpus.vocab,\
            alpha = learning_rate)


    


    start = timeit.default_timer()
    model.fit(iterations = n_gibbs_iteration)         
    stop = timeit.default_timer()
    print('Time for fitting the model is: ', (stop - start)/60) 




    """ printing the top words (results) of corresponding topic most_similar """
    for k in range(n_topic):
    
        """ top words acc. to similairty to positive direction of cosine (standard direction) """
        print("TOPIC {0} w2v    pos:".format(k), \
             list(zip(*model.words2vec.most_similar( positive = [model.topic_params[k]["Topic Mean"]], topn = 20) ))[0])
        
        
        if pre_trained_embedding:
            print('-----------------------------------------------------------------------------------------------------')
            """ in case u r intrested to have top words acc. similairty to an embedding that contains only words in the corpus """
            print("TOPIC {0} w2v_shrinked pos:".format(k), \
                  list(zip(*shrinkedEmbedding_OfWord2vecFormat.most_similar( positive = [model.topic_params[k]["Topic Mean"]], topn = 20) ))[0])
        
        print('\n')
        """ OBS: We assume that the true labels in Amazon reviews are: {'books', 'camera', 'dvd', 'software', 'music', 'other product such as health'} """



