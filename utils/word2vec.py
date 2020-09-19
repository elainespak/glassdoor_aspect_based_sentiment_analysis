#!/usr/bin/env python
# coding: utf-8

import torch
import gensim
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim.downloader as api
from aspect_matching import cosine
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
pd.set_option('display.max_columns', 50)


class Word2VecEmbeddings:
    
    def __init__(self, sentences):
        self.sentences = sentences
        
    def get_pretrained_model(self, model_type):
        self.pretrained_model = api.load(model_type)
        vocab = list(self.pretrained_model.vocab.keys())
        print(f" Loaded vocab size {len(vocab)} from {model_type}")
    
    def train_custom_model(self, model_type):
        self.model = gensim.models.Word2Vec(self.sentences,
                                            size=200, # Word embedding dimension
                                            window=5, # Context window for words during training
                                            min_count=50, # Ignore words that appear less than this
                                            workers=4) # Number of processors (parallelisation)
        print(" Done training")
    
    def most_similar(self, word, n):
        print('\n Pre-trained model says:')
        self.pretrained_similar = self.pretrained_model.wv.most_similar(word, topn = n)
        print(self.pretrained_similar)
        print('\n Our model says:')
        self.model_similar = self.model.wv.most_similar(word, topn = n)
        print(self.model_similar)
        
        

if __name__ == "__main__":
    
    # Call data
    origin = torch.load('../sample_data/master/review_metadata.pt')
    origin = origin[['reviewId','company']]
    sentence = torch.load('../sample_data/master/sentence_match.pt')    
    final = pd.merge(sentence, origin, on='reviewId')
    del origin, sentence
    
    # Train
    text_type = 'pros'
    final = final[final['textType']==text_type][['trigramSentence', 'company']]
    
    doc2vec_corpus = []
    for words, company in tqdm(zip(final['trigramSentence'], final['company'])):
        doc2vec_corpus.append(TaggedDocument(words, [company]))
    doc2vec_model = Doc2Vec(doc2vec_corpus)
    len(doc2vec_model.docvecs)
    for idx, doctag in sorted(doc2vec_model.docvecs.doctags.items(), key=lambda x:x[1].offset):
        print(idx, doctag)
        
    doc2vec_model.docvecs.most_similar('Amazon_com_Inc.')
    
    
    
    sentences = list(master['trigramSentence'])
    model = gensim.models.Word2Vec(sentences,
                                   size=200, # Word embedding dimension
                                   window=5, # Context window for words during training
                                   min_count=50, # Ignore words that appear less than this
                                   workers=4) # Number of processors (parallelisation)
    print(model.wv.most_similar('pay', topn = 40))
    
    #d = {i: np.mean(model.wv[s], axis=0) for i, s in zip(master['sentenceId'], master['trigramSentence'])}
    
    test1 = np.mean(model.wv[sentences[0]],axis=0)
    test2 = np.mean(model.wv[sentences[13]], axis=0)
    test3 = np.mean(model.wv[sentences[8]],axis=0)
    test4 = np.mean(model.wv['flexible', 'work', 'hour'],axis=0)
    cosine(test1, test4)
    print('Model further trained on Glassdoor says:')
    print(word2vec_model.most_similar('agility', topn = 40))
    #model_file = '../sample_data/abae/'+text_type+'/w2v_embedding'
    #model.save(model_file)
    
    
    