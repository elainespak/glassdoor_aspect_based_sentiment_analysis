# -*- coding: utf-8 -*-

import re
import nltk
import torch
import itertools
import pandas as pd
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
pd.set_option('display.max_columns', 500)


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
    only_text = master[text_type]
    only_text['all'] = ''
    for i in range(len(text_type)):
        only_text['all'] += only_text[text_type[i]] + '. '
    
    return list(only_text['all'])


def preprocess_word_tokenize(raw_sentences):
    """
    ###  INPUT
    # raw_sentences: list of raw sentences
    ###  OUTPUT
    # tokenized_sentences: list of processed sentences
    """
    count = len(raw_sentences)
    tokenized_sentences = []
    keep_track = 0
    
    for raw in raw_sentences:
        
        # Change for proper sentence tokenization
        raw = re.sub('\r\n|\n-|\n|\r','. ', raw)
        raw = re.sub(',\.+ ', ', ', raw)
        raw = re.sub('\.+ ', '. ', raw)
        raw = re.sub('&amp;', '&', raw)
        
        # Sentence tokenization
        sentences = sent_tokenize(raw)
        
        # Lowercase, remove punctuations, remove stopwords
        temp = []
        for sent in sentences:
            sent_token = [w for w in CountVectorizer().build_tokenizer()(sent.lower())]
            sent_rmstop = [w for w in sent_token if w not in stop]
            sent = [lmtzr.lemmatize(w) for w in sent_rmstop]
            temp.append(sent)
        tokenized_sentences.append(temp)
        keep_track += 1
        if keep_track % 5000 == 0:
            print(f'{keep_track}/{count} done!')
            print(temp)
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
    
    # Parameters
    path = '../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/'
    text_list = ['summary','pros','cons','advice']
    
    # Tokenize each review into sentence, and then into words
    master = torch.load(path + 'english_glassdoor_reviews.pt')
    for text_type in text_list:
        raw_sentences = list(master[text_type])
        tokenized_sentences = preprocess_word_tokenize(raw_sentences)
        master[text_type+'_tokenized'] = pd.Series(tokenized_sentences)
        print(f'\n *** {text_type} done! ***\n')
    print('----------------------------------------------  Done with tokenization!')
    
    # Make bigram and trigram models
    all_tokenized_sentences = []
    for col in master:
        if col.endswith('_tokenized'):
            all_tokenized_sentences += list(master[col])
    flat = [to_one_list(a) for a in all_tokenized_sentences]
    b_model, t_model = make_ngrams_model(flat, 20, 80)
    torch.save(b_model, path + 'english_glassdoor_reviews_bigram_model.pt')
    torch.save(t_model, path + 'english_glassdoor_reviews_trigram_model.pt')
    print('---------------------------------------  Done with making n-gram model!')
    
    # Turn words into bigrams and trigrams, and then stem them
    for col in master:
        if col.endswith('_tokenized'):
            temp = list(master[col])
            new = [make_ngrams(b_model, t_model, review) for review in temp]
            master[col+'_bigram'] = [n[0] for n in new]
            master[col+'_trigram'] = [n[1] for n in new]
            #master[col+'_bigram_stemmed'] = [stemming(n[0]) for n in new]
            #master[col+'_trigram_stemmed'] = [stemming(n[1]) for n in new]
    print('-------------------------------------------  Done with ngram & stemming!')
    
    # Save
    torch.save(master, path + 'english_glassdoor_reviews_text_preprocessed.pt')
    print('----------------------------------------------------------  Done saving!')
    
    # Create vocabs list and vocabs dictionary
    bigram_sentences, trigram_sentences = make_ngrams(b_model, t_model, tokenized_sentences)    
    vocab, vocab_dict = create_vocab(trigram_sentences)
    print(' *** Check the quality of n-grams:')
    print([v for v in vocab if '_' in v])
    
    # Save
    torch.save(vocab, path + 'english_glassdoor_reviews_vocab.pt')
    torch.save(vocab_dict, path + 'english_glassdoor_reviews_vocab_dict.pt')
    print('--------------------------- Finished saving vocabs -------------------------------------')
    
    
    
    