# -*- coding: utf-8 -*-

import re
import json
import langid
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from gensim.models.phrases import Phrases, Phraser
stemmer = PorterStemmer()


def load_file(file, text='pros'):    
    f = json.load(open(file))
    f = list(f.values())
    temp = []
    if type(text) != list:
        for r in f:
            try:
                if langid.classify(r.get(text))[0]=='en':
                    temp.append((r.get(text), r.get('ratingOverall')))
            except:
                print('error with langdetect')
    else:
        for r in f:
            text_concat = []
            try:
                is_all_english = [langid.classify(r.get(t))[0] for t in text]
                if list(set(is_all_english)) == ['en']:
                    for t in text:
                        text_concat.append(r.get(t))
                    text_all = '. '.join(text_concat)
                    temp.append((text_all, r.get('ratingOverall')))
                # Not all parts are in English, so discard the review
                else:
                    pass
            except:
                pass
    reviews = [t[0] for t in temp]
    ratings = [t[1] for t in temp]
    return reviews, ratings


def parse_all_reviews_to_sentence(reviews, remove_company_list, replace_punctuation):
    """
    # review_processed: list of review, which is a list of processed sentences
    # raw: original review, which is a list of sentences
    # only_sent: list of processed sentences
    """
    review_processed = []
    raw = []
    only_sent = []
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
            s = [w for w in word_tokenize(s) if not w in stopwords]
            sent.append(s)
        review_processed.append(sent)
        only_sent.extend(sent)
    return review_processed, raw, only_sent


def make_ngrams_model(tokenized_sents, set_min_count=5, set_threshold=20):
    bigram = Phrases(tokenized_sents, min_count=set_min_count, threshold=set_threshold)
    trigram = Phrases(bigram[tokenized_sents], threshold=set_threshold)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    return bigram_mod, trigram_mod


def stemming(tokenized_sents):
    sentences = []
    for sent in tokenized_sents:
        stemmed = [stemmer.stem(w) for w in sent]
        if len(stemmed)>0:
            sentences.append(stemmed)
    return sentences

