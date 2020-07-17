#!/usr/bin/env python
# coding: utf-8

# Import necessary packages
import pickle
import gensim.downloader as api
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phrases, Phraser


def load_word2vec(model_type):
    """ Load pre-trained model vectors """
    wv_from_bin = api.load(model_type)
    vocab = list(wv_from_bin.vocab.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


if __name__ == "__main__":
    
    # Pre-trained model
    model_type = 'word2vec-google-news-300'#'glove-wiki-gigaword-100'
    pretrained_model = load_word2vec(model_type)
    
    # Pre-trained model trained on the Glassdoor corpus
    with open('../sample_output/preprocessed_sentences_1gram.pkl', 'rb') as fp:
        sentences = pickle.load(fp)
    word2vec_model = Word2Vec(sentences, 
                              min_count=3,# Ignore words that appear less than this
                              size=300,# Word embedding dimension
                              workers=2,# Number of processors (parallelisation)
                              window=5,# Context window for words during training
                              iter=30# Number of epochs training over corpus
                              ) 
    
    print(f'Pre-trained model {model_type} says:')
    print(pretrained_model.most_similar('people', topn = 10))
    print('Model further trained on Glassdoor says:')
    print(word2vec_model.most_similar('people', topn = 10))
