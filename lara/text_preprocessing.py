# -*- coding: utf-8 -*-

import re
import nltk
import torch
import itertools
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.phrases import Phrases, Phraser
from nltk import word_tokenize, sent_tokenize, FreqDist

stemmer = PorterStemmer()
nltk.download('stopwords')


def to_one_list(lists):
    """ list of lists to one list , e.g. [[1,2],[3,4]] -> [1,2,3,4] """
    return list(itertools.chain.from_iterable(lists))


def load_only_text(pt_file, text_type=['summary', 'pros', 'cons', 'advice'], company=False):
    
    master = torch.load(pt_file)
    if company==False:
        pass
    else:
        master = master[master['company']==company]
    
    # Combine all texts regardless of text_type
    only_text = master[text_type]
    only_text['all'] = ''
    for i in range(len(text_type)):
        only_text['all'] += only_text[text_type[i]] + '. '
    
    return list(only_text['all'])


def parse_all_reviews_to_sentence(raw_reviews, replace_punctuation):
    """
    ###  INPUT
    # raw_reviews: list of raw sentences
    ###  OUTPUT
    # processed_reviews: list of processed sentences
    """
    count = len(raw_reviews)
    processed_reviews = []
    keep_track = 0
    for raw in raw_reviews:
        raw = re.sub('\r\n|\n-|\n|\r','. ', raw)
        raw = re.sub(',\.+ ', ', ', raw)
        raw = re.sub('\.+ ', '. ', raw)
        raw = re.sub('&amp;', '&', raw)
        
        sentences = sent_tokenize(raw)
        sentences = [sent for sent in sentences if len(sent) != 1]
        
        if keep_track % 50 == 0:
            print('PROCESSED: ')
            print(sentences)
        # Lowercase, remove punctuations, remove stopwords
        sent = []
        for s in sentences:
            s = s.lower()
            s = s.translate(replace_punctuation)
            s = [w for w in word_tokenize(s) if not w in stopwords.words('english')]
            sent.append(s)
        processed_reviews.extend(sent)
        keep_track += 1
        if keep_track % 30000 == 0:
            print(f'{keep_track}/{count} done!')
    return processed_reviews


def make_ngrams_model(tokenized_sents, set_min_count=5, set_threshold=20):
    bigram = Phrases(tokenized_sents, min_count=set_min_count, threshold=set_threshold)
    trigram = Phrases(bigram[tokenized_sents], threshold=set_threshold)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    return bigram_mod, trigram_mod


def make_ngrams(bigram_mod, trigram_mod, tokenized_sents):
    b = [bigram_mod[sent] for sent in tokenized_sents]
    t = [trigram_mod[bigram_mod[sent]] for sent in tokenized_sents]
    return b, t


def stemming(tokenized_sents):
    sentences = []
    for sent in tokenized_sents:
        stemmed = [stemmer.stem(w) for w in sent]
        if len(stemmed)>0:
            sentences.append(stemmed)
    return sentences


def create_vocab(sent):
    # Gather every word from every sentence into one list
    words = to_one_list(sent)
    # Count occurrence of every word
    freq = FreqDist(words)
    # Create the official "vocab" with only frequent words
    vocab = [k for k,v in freq.items() if v > 0]
    # Assign a special unique number corresponding to each word
    vocab_dict = dict(zip(vocab, range(len(vocab))))
    return vocab, vocab_dict


if __name__ == "__main__":
    
    import string
    maketrans = ''.maketrans
    replace_punctuation = maketrans(string.punctuation, ' '*len(string.punctuation))
    
    test = load_only_text(r'C:\Users\elain\Desktop\glassdoor_aspect_based_sentiment_analysis\sample_data\2008 to 2018 SnP 500 Firm Data_Master English Files\english_glassdoor_reviews.pt')
    test2 = test[:20000]
    test3 = parse_all_reviews_to_sentence(test2, replace_punctuation)
    