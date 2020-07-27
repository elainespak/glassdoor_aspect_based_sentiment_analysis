# -*- coding: utf-8 -*-

import re
import nltk
import torch
import itertools
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.phrases import Phrases, Phraser
from nltk import word_tokenize, sent_tokenize, FreqDist

stemmer = PorterStemmer()
nltk.download('stopwords')


def to_one_list(lists):
    """ list of lists to one list , e.g. [[1,2],[3,4]] -> [1,2,3,4] """
    return list(itertools.chain.from_iterable(lists))


def load_only_text(pt_file, text_type=['summary','pros','cons','advice'], company=False):

    master = torch.load(pt_file)
    if company==False:
        pass
    else:
        # if company is specified, filter for only that company's review data
        master = master[master['company']==company]
    
    # Combine all texts regardless of text_type
    only_text = master[text_type]
    only_text['all'] = ''
    for i in range(len(text_type)):
        only_text['all'] += only_text[text_type[i]] + '. '
    
    return list(only_text['all'])


def preprocess_word_tokenize(raw_sentences, replace_punctuation):
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
        #if keep_track % 50 == 0:
        #    print('PROCESSED: ')
        #    print(sentences)
        
        # Lowercase, remove punctuations, remove stopwords
        temp = []
        for sent in sentences:
            sent = sent.lower()
            sent = sent.translate(replace_punctuation)
            sent = [w for w in word_tokenize(sent) if w not in stopwords.words('english') and w != []]
            temp.append(sent)
        tokenized_sentences.extend(temp)
        keep_track += 1
        if keep_track % 10000 == 0:
            print(f'{keep_track}/{count} done!')
            print(temp)
    return tokenized_sentences


def make_ngrams_model(tokenized_sentences, set_min_count=5, set_threshold=100):
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
    words = to_one_list(stemmed_sents)
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
    
    test = load_only_text(r'C:\Users\elain\Desktop\glassdoor_aspect_based_sentiment_analysis\sample_data\2008 to 2018 SnP 500 Firm Data_Master English Files\english_glassdoor_reviews_sample.pt')
    tokenized_sentences = preprocess_word_tokenize(test, replace_punctuation)
    
    import pickle
    with open (r'C:\Users\elain\Desktop\glassdoor_aspect_based_sentiment_analysis\sample_data\processed_sentences_sample.pkl', 'wb') as f:
        pickle.dump(tokenized_sentences, f)
    
    import pickle
    with open (r'C:\Users\elain\Desktop\glassdoor_aspect_based_sentiment_analysis\sample_data\processed_sentences_sample.pkl', 'rb') as f:
        tokenized_sentences = pickle.load(f)
        
    b_model, t_model = make_ngrams_model(tokenized_sentences, 5, 100)
    bigram_sentences, _ = make_ngrams(bigram_mod=b_model, trigram_mod=t_model, tokenized_sents=tokenized_sentences)
    stemmed_sentences = stemming(bigram_sentences)
    vocab, vocab_dict = create_vocab(stemmed_sentences)
    
    ### ------------------------------------ Check how frequent the bigrams are!
    test = to_one_list(stemmed_sentences)
    test_freq = FreqDist(test)
    ok = sorted([(test_freq[k],k) for k,v in vocab_dict.items() if '_' in k], reverse=True)
    print(ok[:15])
    ### ------------------------------------------------------------------------
    