# -*- coding: utf-8 -*-


import re
import nltk
import torch
import itertools
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk import sent_tokenize, FreqDist
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('wordnet')
nltk.download('stopwords')
stop = stopwords.words('english')

stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()
pd.set_option('display.max_columns', 50)


def to_one_list(lists):
    """ list of lists to one list , e.g. [[1,2],[3,4]] -> [1,2,3,4] """
    return list(itertools.chain.from_iterable(lists))


def load_only_text(master, text_type, company=False):

    if company==False:
        pass
    else:
        # if company is specified, filter for only that company's review data
        master = master[master['company']==company]
        print(f'{company} text loaded!')
    
    # Combine all texts regardless of text_type
    all_tokenized_sentences = []
    for t in text_type:
        all_tokenized_sentences += list(master[t])
    
    return all_tokenized_sentences


def preprocess_word_tokenize(original_sentence):
    """preprocess a single sentence to a list of word tokens"""
    sent_token = [w for w in CountVectorizer().build_tokenizer()(original_sentence.lower())]
    sent_rmstop = [w for w in sent_token if w not in stop]
    tokenized_sentences = [lmtzr.lemmatize(w) for w in sent_rmstop]
    return tokenized_sentences


def make_ngrams_model(tokenized_sentences, set_min_count=30, set_threshold=80):
    bigram = Phrases(tokenized_sentences, min_count=set_min_count, threshold=set_threshold)
    trigram = Phrases(bigram[tokenized_sentences], threshold=set_threshold)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    return bigram_mod, trigram_mod


def make_ngrams(bigram_mod, trigram_mod, tokenized_sents):
    bigram_sents = [bigram_mod[sent] for sent in tokenized_sents]
    trigram_sents = [trigram_mod[bigram_mod[sent]] for sent in tokenized_sents]
    return bigram_sents, trigram_sents


def stemming(ngram_sents):
    stemmed_sents = []
    for sent in ngram_sents:
        stemmed = [stemmer.stem(w) for w in sent]
        if len(stemmed)>0:
            stemmed_sents.append(stemmed)
    return stemmed_sents


def create_vocab(stemmed_sents):
    # Gather every word from every sentence into one list
    words = to_one_list(to_one_list(stemmed_sents))
    # Count occurrence of every word
    freq = FreqDist(words)
    # Create the official "vocab" with only frequent words
    vocab = [k for k,v in freq.items() if v > 50]
    # Assign a special unique number corresponding to each word
    vocab_dict = dict(zip(vocab, range(len(vocab))))
    return vocab, vocab_dict


if __name__ == "__main__":
    # Load master review metadata
    path = '../sample_data/master/'
    master = torch.load(path + 'review_metadata.pt')
    
    # Parameters of interest
    text_list = ['pros','cons','advice']
    
    # Explode to sentence level    
    match = pd.melt(master, id_vars=['reviewId'], value_vars=text_list)
    match = match.explode('value')
    match.columns = ['reviewId', 'textType', 'originalSentence']
    match = match[~pd.isnull(match['originalSentence'])]
    match = match[match['originalSentence'].str.len()>1]
    match = match.reset_index(drop=True)
    match['sentenceId'] = match.index.values
    del master
    
    # Tokenize with the trigram model
    word_tokens = []
    for sentence in tqdm(match['originalSentence']):
        word_tokens.append(preprocess_word_tokenize(sentence))
    
    match['tokenizedSentence'] = word_tokens
    match['length'] = match['tokenizedSentence'].apply(lambda l: len(l))
    match = match[match['length'] > 1]
    match = match.drop(columns = ['length'])
    del word_tokens
    
    b_model, t_model = make_ngrams_model(list(match['tokenizedSentence']), 20, 80)
    torch.save(b_model, path + 'bigram_model.pt')
    torch.save(t_model, path + 'trigram_model.pt')
    
    _, match['trigramSentence'] = make_ngrams(b_model, t_model, match['tokenizedSentence'])
    torch.save(match, path + 'sentence_match.pt')
    
    # Create vocabs list and vocabs dictionary
    vocab, vocab_dict = create_vocab(list(match['trigramSentence']))
    torch.save(vocab, path + 'vocab.pt')
    torch.save(vocab_dict, path + 'vocab_dict.pt')
    