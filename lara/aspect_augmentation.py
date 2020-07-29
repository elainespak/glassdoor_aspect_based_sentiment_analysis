# -*- coding: utf-8 -*-

import os
import json
import math
import numpy as np
from nltk import FreqDist
from text_preprocessing import *
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def label_sentence_UseVocab(final_sentences, VocabDict):
    '''
    Label every word of a sentence by using:
    1) a corresponding number from the VocabDict (vocabulary lookup table)
        (e.g., '-Amazing people' --> [0, 1]) which really means ['amaz', 'peopl'])
        OR
    2) "None" label (TODO)
    '''
    vocabdict_labeled_sentences = []
    for sentence in final_sentences:
        temp = [VocabDict[word] for word in sentence]
        if len(temp)>0:
            vocabdict_labeled_sentences.append(temp)
    return vocabdict_labeled_sentences


class Sentence_info:
    def __init__(self, vocabdict_labeled_sentence):
        '''
        ###  INPUT
        # sent: vocabdict_labeled_sentence
          ( e.g., '-Amazing people' --> [0, 1]) which really means ['amaz', 'peopl'] )
        
        ###  DEFINED
        # word_frequency: occurrence count of each word in sent
        # label: initially, set to -1
        '''
        self.word_frequency = FreqDist(vocabdict_labeled_sentence)
        self.aspect_label = -1


class Review:
    def __init__(self, review_text, VocabDict, text_type):
        '''
        ###  INPUT
        # review_text: review text
        # VocabDict: vocabulary lookup table
        '''
        tokenized_sentences = preprocess_word_tokenize(review_text, replace_punctuation)
        bigram_sentences, _ = make_ngrams(b_model, t_model, tokenized_sentences) # Todo: make it possible to choose bi- or tri-
        stemmed_sentences = stemming(bigram_sentences)
        
        vocabdict_labeled_sentences = label_sentence_UseVocab(stemmed_sentences, VocabDict)
        
        self.Sentences_info = [Sentence_info(sent) for sent in vocabdict_labeled_sentences]
        UniWord = {} # dictionary
        for sent in self.Sentences_info:
            UniWord = UniWord | sent.word_frequency.keys() # now, UniWord is a set
        #UniWord = {-1 if w == None else w for w in UniWord}
        
        self.UniWord = np.array([w for w in UniWord])
        self.UniWord.sort()
        self.NumOfUniWord = len(self.UniWord)


class Company:
    def __init__(self, path, company_name, VocabDict, text_type):
        '''
        ###  INPUT
        # company_name: company name ( e.g., 'XYZ International Inc.' )
        '''
        self.Reviews = [Review(text, VocabDict, text_type) for text in load_only_text(path, company_name, text_type)]
        self.NumOfReviews = len(self.Reviews)


class Corpus:
    def __init__(self, path, corpus, Vocab, VocabDict, text_type):
        '''
        ###  INPUT
        # corpus: list of all companies
        '''
        self.Companies = [Company(path, company, VocabDict, text_type) for company in corpus]
        self.NumOfCompanies = len(corpus)
        self.VocabLength = len(Vocab)
        self.Aspect_Terms = []
        
        
def label_aspect(Sentence_info, aspects_num, K):
    '''
    ###  INPUT
    # aspects_num: list of list of aspects
                   ( e.g., [['pay','money','benefits'], ['coworkers','team','colleagues']] )
    # K: number of different aspects
    
    ###  OUTPUT
    # match_count: k-dimensional vector representing the number of aspect words in the review
    '''
    match_count = np.zeros(K)
    for idx in range(K):
        for word_num, word_num_count in Sentence_info.word_frequency.items():
            if word_num in aspects_num[idx]:
                match_count[idx] += word_num_count
    return match_count



def ChisqTest(N, taDF, tDF, aDF):
    '''
    ###  INPUT
    # N: all sentences that have some sort of aspect label
    # taDF: term in the aspect-labeled Document Frequency
    # tDF: term Document Frequency
    # aDF: aspect-labeled Document Frequency
    '''
    A = taDF  ## term & aspect
    # A+B = tDF
    B = tDF - A # term occurring in non-aspect Document Frequency
    C = aDF - A # number of sentences without the term
    D = N - A - B - C 
    return ((N * ( A * D - B * C )**2)) / ((aDF * ( B + D ) * tDF * ( C + D )) + 0.00001)


def collect_stat_for_each_review(review, aspects, Vocab):
    '''
    ###  INPUT
    # review: each review
    # aspects: list of list of aspects
               ( e.g., [[11, 48, 4], [4, 2, 29]], which represent
               [['pay','money','benefits'], ['coworkers','team','colleagues']] )
    '''
    # review.num_stn_aspect_word = np.zeros((len(aspect),len(Vocab)))
    K = len(aspects)
    review.num_stn_aspect_word = np.zeros((K,review.NumOfUniWord))
    review.num_stn_aspect = np.zeros(K)
    review.num_stn_word = np.zeros(review.NumOfUniWord)
    review.num_stn = 0
    
    for Sentence in review.Sentences_info:
        if Sentence.aspect_label != -1:  # if the sentence has an aspect label,
            review.num_stn += 1
            for l in Sentence.aspect_label:
                review.num_stn_aspect[l] += 1
            for w,v in Sentence.word_frequency.items():#keys():
                z = np.where(w == review.UniWord)[0]  # index
                review.num_stn_word[z] += v
            for l in Sentence.aspect_label:
                for w,v in Sentence.word_frequency.items():#keys():
                    z = np.where(w == review.UniWord)[0] # index
                    review.num_stn_aspect_word[l,z] += v # FIX HERE???
    return review.num_stn_aspect_word, review.num_stn_aspect, review.num_stn_word, review.num_stn

    
class Bootstrapping:
    
    def sentence_label(self, corpus, K): # produce a label list
        if 0 < K:
            for company in corpus.Companies:
                for text in company.Reviews:
                    for Sentence_info in text.Sentences_info:
                        match_count = label_aspect(Sentence_info, self.Aspect_Terms, K)
                        highest_match_count = np.max(match_count)
                        if 0 < highest_match_count: # if at least one of the aspects has a match
                            aspect_label = np.where(highest_match_count==match_count)[0].tolist() # index of the aspect with max matched terms
                            Sentence_info.aspect_label = aspect_label # TODO: how about a tie?
                        else:
                            pass
        else:
            print("Warning: No sentences or Aspect_Terms are recorded in this corpus")

    def calc_chi_sq(self, corpus, K):
        V = corpus.VocabLength
        corpus.all_num_stn_aspect_word = np.zeros((K,V))
        corpus.all_num_stn_aspect = np.zeros(K)
        corpus.all_num_stn_word = np.zeros(V)
        corpus.all_num_stn = 0
        Chi_sq = np.zeros((K,V))
        
        if 0 < K:
            for company in corpus.Companies:
                for text in company.Reviews:
                    text.num_stn_aspect_word, text.num_stn_aspect, text.num_stn_word, text.num_stn = collect_stat_for_each_review(text, self.Aspect_Terms, corpus.Vocab)
                    corpus.all_num_stn += text.num_stn # total number of sentences with any aspect label
                    corpus.all_num_stn_aspect += text.num_stn_aspect
                    for w in text.UniWord:
                        z = np.where(w == text.UniWord)[0][0] # index, since the matrix for review text is small
                        corpus.all_num_stn_word[w] += text.num_stn_word[z] # number of times aspect_i words (z) appear in all sentences
                        corpus.all_num_stn_aspect_word[:,w] += text.num_stn_aspect_word[:,z]

            for k in range(K):
                for w in range(V):
                    Chi_sq[k,w] = ChisqTest(
                            corpus.all_num_stn, # sentences with any aspect
                            corpus.all_num_stn_aspect_word[k,w], # num. of words in sentences belonging to aspect_k
                            corpus.all_num_stn_word[w], # num. of word occurrence in any sentences
                            corpus.all_num_stn_aspect[k] # num. of sentences of aspect_k
                            )
            self.Chi_sq = Chi_sq
        else:
            print("Warning: No aspects were pre-specified")

def load_Aspect_Terms(analyzer, seedwords_path, VocabDict):
    ''' 
    # seedwords_path: path where the aspect seedwords text file is located
    # VocabDict: vocab lookup table
    '''    
    analyzer.Aspect_Terms=[]
    with open(seedwords_path, 'r') as f:
        for line in f:
            aspect = [VocabDict[stemmer.stem(w.strip().lower())] for w in line.split(',')]
            analyzer.Aspect_Terms.append(aspect)
    print("---------------------------- Aspect Keywords loading completed! ---------")

def Add_Aspect_Keywords(analyzer, p, NumIter, c):
    '''
    ###  INPUT
    # analyzer
    # p: Maximum number of added aspect words in each round
    # NumIter: Maximum number of iterations
    # c: data
    '''
    for i in range(NumIter):
        analyzer.sentence_label(c)
        analyzer.calc_chi_sq(c)
        t = 0
        for cs in analyzer.Chi_sq:
            x = cs[np.argsort(cs)[::-1]] # descending order
            y = np.array([not math.isnan(v) for v in x]) # return T of F
            words = np.argsort(cs)[::-1][y] #
            aspect_num = 0
            for w in words:
                if w not in to_one_list(analyzer.Aspect_Terms):
                    analyzer.Aspect_Terms[t].append(w)
                    aspect_num += 1
                if aspect_num > p:
                    break
            t += 1
        print(" *** Complete iteration "+str(i+1)+"/"+str(NumIter))


def save_Aspect_Keywords_to_file(analyzer, finalwords_path, Vocab):
    '''
    ###  INPUT
    # finalwords_path: path where the complete aspect words text file will locate
    '''
    with open(finalwords_path, 'w') as f:
        for aspect in analyzer.Aspect_Terms:
            for w in aspect:
                try:
                    f.write(Vocab[w]+", ")
                except:
                    pass
            f.write("\n\n\n")
            

def create_W_matrix_for_each_review(analyzer, review, corpus):
    Nd = len(review.UniWord)
    K = len(analyzer.Aspect_Terms)
    # V=len(corpus.Vocab)
    review.W = np.zeros((K, Nd))
    for k in range(K):
        for w in range(Nd):  ## w is index of UniWord_for_review
            # z = review.UniWord[w]
            if corpus.all_num_stn_aspect[k] > 0:
                review.W[k,w] = review.num_stn_aspect_word[k,w] / corpus.all_num_stn_aspect[k]


def create_all_W(analyzer, corpus):
    company_num=0
    for company in corpus.Companies:
        print("Creating W matrix for company '"+str(company_num)+"': "+company.Company)
        for review in company.Reviews:
            create_W_matrix_for_each_review(analyzer, review, corpus)
        company_num += 1


def produce_data_for_rating(analyzer, corpus, outputfolderpath, percompany=False):
    dir = outputfolderpath
    if not os.path.exists(dir):
        os.makedirs(dir)

    vocabfile = outputfolderpath + "vocab.txt"
    f = open(vocabfile,"w",encoding='UTF-8')
    for w in corpus.Vocab:
        f.write(w+",")
    f.close()
    
    vocabdictfile = outputfolderpath + "vocab_dict.txt"
    with open(vocabdictfile,"w",encoding='utf8') as f:
        json.dump(corpus.VocabDict, f)
    '''
    json.dumps(corpus.VocabDict)
    for w in corpus.VocabDict.items():
        f.write(w)
        f.write("\n")
    '''
    f.close()
    
    if percompany==False:
        reviewfile = outputfolderpath + "review_data_all.txt"
        f = open(reviewfile, 'w', encoding='utf8')
        for company in corpus.Companies:
            for review in company.Reviews:
                f.write(str(review.reviewId))
                f.write(":")
                f.write(str(review.Overall))
                f.write(":")
                f.write(str(review.UniWord.tolist()))
                f.write(":")
                f.write(str(review.W.tolist()))
                f.write("\n")
            f.write("\n")
        f.close()
        
    else:
        reviewfile = outputfolderpath + "review_data.txt"
        f = open(reviewfile, 'w', encoding='utf8')
        for company in corpus.Companies:
            f.write(company.Company)
            f.write("\nTotal number of reviews: " + str(company.NumOfReviews))
            f.write("\n")
            for review in company.Reviews:
                f.write(str(review.reviewId))
                f.write(":")
                f.write(str(review.Overall))
                f.write(":")
                f.write(str(review.UniWord.tolist()))
                f.write(":")
                f.write(str(review.W.tolist()))
                f.write("\n")
            f.write("\n")
        f.close()