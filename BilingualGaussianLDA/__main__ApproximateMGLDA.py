import time
import numpy as np
import os
from gensim.models import FastText
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import corpus as nltk_corpus
from filtering import *
from corpus import Corpus
from Approximate_fasttextMGLDA import *




def read_documents(doc_file):
    docs = []; labels = []; 
    with open(doc_file, encoding = 'utf-8') as file:  # encoding
        for line in file:
            label, _, _, doc = line.strip().split( maxsplit = 3 )
            docs.append(doc)
            labels.append(label)
    return docs, labels


def read_documents_novel(urpath, novel_name):
    docs = []
    for subdir, dirs, files in os.walk( urpath + 'data/other_data/'):
        for i in range(len(files)):
            filepath = subdir + os.sep + files[i]
            if filepath.endswith( novel_name ):
                with open(filepath, encoding = 'utf-8') as f:
                    contents = f.read()
                    docs.append(contents)
    return docs




""" Define ur path there u have the folders of the data """
urpath = 'C:/Users/elias/OneDrive/Skrivbord/github/thesis/'


"""  Define ur source and target language  """
source_language = 'english'
target_language = 'swedish'
stemmer_source = SnowballStemmer(source_language)
stemmer_target = SnowballStemmer(target_language)  
stop_words_source = set(nltk_corpus.stopwords.words(fileids = source_language))
stop_words_target = set(nltk_corpus.stopwords.words(fileids = target_language))


""" in case the corpus is Amazon reviews or other files such as novels etc. """
Amazon_reviews = True

# u can change the hyper-parameters to the ones in the thesis in order to get similar results.
rate_usageOfData_source_Amazon = 0.01
rate_usageOfData_target_Amazon = 0.1
rate_usageOfData_source_Novel = 1.0
rate_usageOfData_target_Novel = 1.0
learning_rate = 0.05
n_topic = 5
n_gibbs_iterations_source = 3
n_gibbs_iterations_target = 2
dim_trained_embedding = 20


if __name__ == "__main__":

    """ step 1. reading and filtering the corppus of the source and target language """
    if Amazon_reviews:
        
        docs_source, labels_source = read_documents(urpath + 'data/Amazon_data/amazon-English-EntireData.txt')
        docs_target, labels_target = read_documents(urpath + 'data/Amazon_data/amazon-Swedish-first10%.txt')

        
    else: 
        
        docs_source = read_documents_novel(urpath, "AFoolFree.txt")
        docs_target = read_documents_novel(urpath, "EnDåreFri.txt")

    f_source = filtering(stemmer_source, stop_words_source)
    filtered_docs_source = f_source.preprocessing(docs_source)
    f_target = filtering(stemmer_target, stop_words_target)
    filtered_docs_target = f_target.preprocessing(docs_target) 
    
    
    
    """ step 2. merging both langauges to build a common word embeddings """

    filtered_docs_merged = []
    filtered_docs_merged.extend(filtered_docs_source)
    filtered_docs_merged.extend(filtered_docs_target)
    
    tockenized_docs = []                     # we need to tockenize the documents to build word vector in Fasttext
    [tockenized_docs.append(doc.split(None)) for doc in filtered_docs_merged]
    joint_modelvw = FastText(tockenized_docs, size = dim_trained_embedding, window=5,workers=4,alpha=0.1,iter=10,sg=1,word_ngrams=5, negative=10)

    print('Step 1 and 2 is finished ')
    
    
    
    
    
    """ Step 3: Now training the model on the source language """
    """ Step 3 can be skipped if u r not intrested of the output of corpus of the source language """
    time_source1 = time.time()

    with open(urpath + "temporary_files/docs_filtered_source.txt", "w") as output:
        for doc in filtered_docs_source:
            output.write('%s\n' % doc) # each doc in a new line     
        
    corpus_source = Corpus()
    

    if Amazon_reviews:
        corpus_source.load_text(urpath + "temporary_files/docs_filtered_source.txt" , valid_split = 1-rate_usageOfData_source_Amazon )
    else: 
        corpus_source.load_text(urpath + "temporary_files/docs_filtered_source.txt" , valid_split = 1-rate_usageOfData_source_Novel )

    corpus_source.process_wordvectors(joint_modelvw)

    words2vec_nyOfWord2vec_format = corpus_source.convert_dictionary_to_words2vec(fname= urpath + 'temporary_files/train_source.txt')

    model_source = GLDA(n_topics = n_topic, corpus = corpus_source.index2doc, words2vec_ny = corpus_source.words2vec_ny, \
                 words2vec = joint_modelvw, vocab_ny = corpus_source.vocab_ny, alpha = learning_rate)
    
    model_source.fit(iterations = n_gibbs_iterations_source)
    
    



    """ Now save the trained model on source language """

    print(' saving the parameters of the model that has been trained on the source langugare for using in target language ')      
    model_source.save_model(filepath = urpath + "temporary_files/")
                            
                            
                     
        

    """ Now training the model on the target language and initialize the count matrix as count matrix of the last iteration of source language """
    time_target1 = time.time()

    with open(urpath + "temporary_files/docs_filtered_target.txt", "w") as output:
        for doc in filtered_docs_target:
            output.write('%s\n' % doc) 
        
    corpus_target = Corpus()

    if Amazon_reviews:
        corpus_target.load_text(urpath + "temporary_files/docs_filtered_target.txt", valid_split = 1-rate_usageOfData_target_Amazon )
    else:
        corpus_target.load_text(urpath + "temporary_files/docs_filtered_target.txt", valid_split = 1-rate_usageOfData_target_Novel )

    corpus_target.process_wordvectors(joint_modelvw)
    words2vec_nyOfWord2vec_format_target = corpus_target.convert_dictionary_to_words2vec(fname = urpath + 'temporary_files/train_target.txt')


    model_target = GLDA(n_topics = n_topic, corpus = corpus_target.index2doc, words2vec_ny = corpus_target.words2vec_ny,\
              words2vec = joint_modelvw, vocab_ny = corpus_target.vocab_ny, alpha = learning_rate)
                            
    
    model_target.load_model(filepath = urpath + 'temporary_files/')

    model_target.fit(iterations = n_gibbs_iterations_target)                           
                            
                          


    for k in range(n_topic):
    
        """ top words acc. to similairty to positive direction of cosine (standard direction) """
        print("TOPIC {0} w2v    pos:".format(k), \
             list(zip(*model_target.words2vec.most_similar( positive = [model_target.topic_params[k]["Topic Mean"]], topn = 20) ))[0])
        print('\n')
        
    time_target2 = time.time()
    print(" The time taken for preproessing of corpus and loading the model for target language is", (time_target2-time_target1)/60)
