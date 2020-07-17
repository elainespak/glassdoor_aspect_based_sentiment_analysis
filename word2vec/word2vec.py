#!/usr/bin/env python
# coding: utf-8

# Import necessary packages
import re
import json
import gensim
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phrases, Phraser



def load_word2vec():
    """ Load pre-trained Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    #wv_from_bin = api.load("word2vec-google-news-300")
    wv_from_bin = api.load("glove-wiki-gigaword-100")
    vocab = list(wv_from_bin.vocab.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin

wv_from_bin = load_word2vec()
wv_from_bin.most_similar("people")[:10]


# Bring in the file
alldat = pd.read_csv('2008_to_2018_SnP500_Names.csv', delimiter=',') # 파일명을 바꾸세요
names = list(alldat['conml']) # 회사명이 적힌 column 이름을 쓰세요
gvkeys = list(alldat['gvkey'])
companies_all = [(names[i],gvkeys[i])for i in range(len(alldat))]
companies_all[:3]

# Read the corpus line by line
def get_sentences(input_file_pointer):
    while True:
        line = input_file_pointer.readline()
        if not line:
            break
        yield line
        
def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    return re.sub(r'\s{2,}', ' ', sentence)

def tokenize(sentence):
    return [token for token in sentence.split() if token not in STOP_WORDS]

# Build bi-grams
def build_phrases(sentences):
    phrases = Phrases(sentences,
                      min_count=5,
                      threshold=7,
                      progress_per=1000)
    return Phraser(phrases)

sentences = []

for name,gvkey in companies_all[:]:
    # Bring in the reviews file
    name = name.replace(' ','_')
    try:
        jsondat = json.load(open('2008 to 2018 SnP 500 Firm Data All/'+name+'_individual_reviews_all.txt'))
    except:
        print(f'No glassdoor data for {name}')
    
    for v in list(jsondat.values()):
        cleaned_sentence = clean_sentence(v['pros'])
        tokenized_sentence = tokenize(cleaned_sentence)
        sentences.append(tokenized_sentence)
        
        cleaned_sentence = clean_sentence(v['cons'])
        tokenized_sentence = tokenize(cleaned_sentence)
        sentences.append(tokenized_sentence)

len(sentences)

model_pros = build_phrases(sentences)


# Add new bi-gram words to the corpus
sentences = list(model_pros[sentences])
sentences[:6]


# Save it
model_pros.save('model_pros_test_year2017.txt')

#Load it
#model_pros= Phraser.load('model_pros.txt')


# Done
word2vec_model = Word2Vec(sentences, 
                 min_count=3,   # Ignore words that appear less than this
                 size=200,      # Dimensionality of word embeddings
                 workers=2,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30)       # Number of epochs training over corpus


word2vec_model.most_similar('opportunities',topn=20)

word2vec_corpus



## lovit
word2vec_model = Word2Vec(
    word2vec_corpus,
    size=100,
    alpha=0.025,
    window=5,
    min_count=5,
    sg=0,
    negative=5)

