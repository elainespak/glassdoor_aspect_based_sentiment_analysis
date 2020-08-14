#!/usr/bin/env python
# coding: utf-8

import torch
import itertools
import gensim.downloader as api
from gensim.models import Word2Vec, Doc2Vec


def load_only_text(master, text_type, company=False):

    if company==False:
        pass
    else:
        # if company is specified, filter for only that company's review data
        master = master[master['company']==company]
        print(f'{company} text loaded!')
    
    # Combine all texts regardless of text_type
    all_sentences = []
    for t in text_type:
        all_sentences += list(master[t])
    
    return all_sentences

def to_one_list(lists):
    """ list of lists to one list , e.g. [[1,2],[3,4]] -> [1,2,3,4] """
    return list(itertools.chain.from_iterable(lists))


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
    
    # Train the pre-trained model on the Glassdoor corpus
    path = 'C:/Users/elain/Desktop/glassdoor_aspect_based_sentiment_analysis/sample_data/'
    master = torch.load(path + '2008 to 2018 SnP 500 Firm Data_Master English Files/english_glassdoor_reviews_text_preprocessed.pt')
    
    all_tokenized_sentences = load_only_text(master,
                                             ['pros_tokenized', 'cons_tokenized', 'advice_tokenized'])
    all_tokenized_sentences = to_one_list(all_tokenized_sentences)
    
    word2vec_model = Word2Vec(all_tokenized_sentences, 
                              min_count=30,# Ignore words that appear less than this
                              size=200,# Word embedding dimension
                              workers=2,# Number of processors (parallelisation)
                              window=5,# Context window for words during training
                              iter=30# Number of epochs training over corpus
                              ) 
    
    print(f'Pre-trained model {model_type} says:')
    print(pretrained_model.most_similar('pay', topn = 40))
    print('Model further trained on Glassdoor says:')
    print(word2vec_model.most_similar('agility', topn = 40))
