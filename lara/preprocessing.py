# -*- coding: utf-8 -*-

import re
import json
import nltk
import fasttext
import itertools
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.phrases import Phrases, Phraser
from nltk import word_tokenize, sent_tokenize, FreqDist

stemmer = PorterStemmer()
nltk.download('stopwords')
lid_model = fasttext.load_model('lid.176.bin') 


def is_english(sentence):
    """ input sentence should be a string """
    if '__label__en' == lid_model.predict(sentence)[0][0]:
        return True
    else:
        return False


def to_one_list(lists):
    """ list of lists to one list , e.g. [[1,2],[3,4]] -> [1,2,3,4] """
    return list(itertools.chain.from_iterable(lists))

def load_file(file, text='pros'):    
    f = json.load(open(file))
    f = list(f.values())
    temp = []
    if type(text) != list:
        for r in f:
            if is_english(r.get(text)) is True:
                temp.append((r.get(text), r.get('ratingOverall')))
    else:
        for r in f:
            length_text = [len(r.get(t)) for t in text]
            idx = length_text.index(max(length_text))
            text_concat = []
            if is_english(r.get(text[idx])) is True:
                for t in text:
                    text_concat.append(r.get(t))
                text_all = '. '.join(text_concat)
                temp.append((text_all, r.get('ratingOverall')))
    reviews = [t[0] for t in temp]
    ratings = [t[1] for t in temp]
    return reviews, ratings


def parse_all_reviews_to_sentence(reviews, remove_company_list, replace_punctuation):
    """
    # review_processed: list of review, which is a list of processed sentences
    # raw: list of review, which is a list of raw sentences
    # only_sent: list of processed sentences
    """
    review_processed = []
    raw = []
    only_sent = []
    keep_track = 0
    for r in reviews:
        sentences = sent_tokenize(r)
        # Often, glassdoor reviews are in a list
        if len(sentences) < 2:
            sentences = re.split('\r\n|\n-',r)
        if len(sentences) < 2:
            sentences = re.split('\r|\n',r)
        raw.append(sentences)
        sent = []
        for s in sentences:
            s = s.lower()
            #remove company mentions in the sentences
            for company_mention in remove_company_list:
                if company_mention in s:
                    s = s.replace(company_mention, '')
			#remove punctuations and stopwords
            s = s.translate(replace_punctuation)
            s = [w for w in word_tokenize(s) if not w in stopwords.words('english')]
            sent.append(s)
        review_processed.append(sent)
        only_sent.extend(sent)
        keep_track += 1
        if keep_track % 5000 == 0:
            print(f'{keep_track}/{len(reviews)} done!')
    return review_processed, raw, only_sent


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

