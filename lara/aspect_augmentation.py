# -*- coding: utf-8 -*-

import os
import json
import math
import langid
import numpy as np
from lara.text_processing import *
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize, word_tokenize, FreqDist
stemmer = PorterStemmer()


def load_review_file(path, company_name, text='pros'):
    filename = company_name.replace(' ', '_')+'_individual_reviews_all.txt'
    file_name = path + filename
    reviews = json.load(open(file_name)).values()
    final_reviews = []
    if type(text) != list:
        for r in reviews:
            if langid.classify(r.get(text))[0]=='en':
                final_reviews.append(r)
    else:
        for r in reviews:
            try:
                is_all_english = [langid.classify(r.get(t))[0] for t in text]
                if list(set(is_all_english)) == ['en']:
                    final_reviews.append(r)
                else:
                    pass # Not all review parts are in English
            except:
                pass
    return final_reviews


def label_sentence_UseVocab(only_sentences, VocabDict):
    '''
    Label every word of a sentence by using:
    1) a corresponding number from the VocabDict (vocabulary lookup table)
        OR
    2) "None" label
    
    '''
    num_sent = []
    for sent in only_sentences:
        temp = [VocabDict.get(w) for w in sent]
        if len(temp)>0:
            num_sent.append(temp)
    return num_sent


class Sentence_info:
    def __init__(self, sent):
        '''
        ###  INPUT
        # sent: num_sent (e.g., '-Amazing people' --> [0, 1]) ###['amaz', 'peopl'])
        
        ###  DEFINED
        # word_frequency: occurrence count of each word in sent 
        # unique_count: count of unique words in sent   < ---- 안쓰임!!!!! 대체 뭐
        # label: initially, set to -1
        '''
        self.word_frequency = FreqDist(sent)
        self.unique_word_count = len(self.word_frequency)
        self.aspect_label = -1


class Review:
    def __init__(self, review_data, VocabDict):
        '''
        ###  INPUT
        # review_data: each individual review
        # VocabDict: vocabulary lookup table
        
        ###  DEFINED
        # Overall: overall rating
        # Sentence_class: class Sentences
        # 
        '''
        self.Overall = review_data.get("ratingOverall")
        # self.WorkLifeBalance = review_data.get("ratingWorkLifeBalance")
        self.reviewId= review_data.get("reviewId")
        # Text
        if type(text_type) != list:
            self.reviewText = review_data.get(text_type)
        else:
            text_concat = []
            for t in text_type:
                text_concat.append(review_data.get(t))
            self.reviewText = '. '.join(text_concat)
        #self.only_sents = parse_to_sentence(self.reviewText) # original
        temp_sents = parse_to_sentence(self.reviewText) # new
        temp_sents, t = make_ngrams(bigram_mod=b_mod, trigram_mod=t_mod, tokenized_sents = temp_sents) # new
        self.only_sents = stemming(temp_sents) # new
        
        num_sents = label_sentence_UseVocab(self.only_sents, VocabDict)
        
        self.Sentences_info = [Sentence_info(num_sent) for num_sent in num_sents]
        UniWord = {}
        for sent in self.Sentences_info:
            UniWord = UniWord | sent.word_frequency.keys()
        UniWord = {-1 if w == None else w for w in UniWord}
        self.UniWord = np.array([w for w in UniWord])
        self.UniWord.sort()
        self.NumOfUniWord = len(self.UniWord)


class Company:
    def __init__(self, company_name, VocabDict):
        '''
        ###  INPUT
        # company_name: company name
                        e.g., 'XYZ International Inc.'
        # company_data: json file itself
        '''
        #self.Company = re.match('.*?(?=_individual_reviews_all.txt)', company_json)
        #self.Reviews = [Review(review, VocabDict) for review in company_data.get("Reviews")]
        self.Company = company_name
        self.Reviews = [Review(review, VocabDict) for review in load_review_file(path, self.Company, text)]
        self.NumOfReviews = len(self.Reviews)


def sent_aspect_match(Sentences_info, aspects, K):
    '''
    ###  INPUT
    # aspects: list of list of aspects
               e.g., [['pay','money','benefits'], ['coworkers','team','colleagues']]
    # k: number of different aspects
    
    ###  OUTPUT
    # match_count: k-dimensional vector representing the number of aspect words in the review
    '''
    match_count = np.zeros(K)
    for idx in range(K):
        for word_num, word_num_count in Sentences_info.word_frequency.items():
            if word_num in aspects[idx]:
                match_count[idx] += word_num_count
    return match_count


class Corpus:
    def __init__(self, corpus, Vocab, VocabDict):
        '''
        ###  INPUT
        # corpus: list of all companies
        # Vocab: all Vocab??????????
        # Count: 
        '''
        self.Vocab = Vocab
        self.VocabDict = VocabDict
        self.V = len(Vocab)
        self.Aspect_Terms = []
        self.Companies = [Company(company_name, self.VocabDict) for company_name in corpus] # rest = one restaurant
        self.NumOfCompanies = len(corpus)


def ChisqTest(N, taDF, tDF, aDF):
    '''
    ###  INPUT
    # N: all sentences that have some sort of aspect label
    # taDF: term in the aspect-labeled Document Frequency
    # tDF: term Document Frequency
    # aDF: aspect-labeled Document Frequency
    Calculate Chi-Square
    '''
    A = taDF  ## term & aspect
    # A+B = tDF
    B = tDF - A # term occurring in non-aspect Document Frequency
    C = aDF - A # number of sentences without the term
    D = N - A - B - C 
    return ((N * ( A * D - B * C )**2)) / ((aDF * ( B + D ) * tDF * ( C + D )) + 0.00001)


def collect_stat_for_each_review(review,aspects,Vocab):
    '''
    ###  INPUT
    # review: each review
    # aspects: list of list of aspects
               e.g., [['pay','money','benefits'], ['coworkers','team','colleagues']]
    '''
    # review.num_stn_aspect_word = np.zeros((len(aspect),len(Vocab)))
    K = len(aspects)
    review.num_stn_aspect_word = np.zeros((K,review.NumOfUniWord))
    review.num_stn_aspect = np.zeros(K)
    review.num_stn_word = np.zeros(review.NumOfUniWord)
    review.num_stn = 0
    
    for Sentence in review.Sentences_info:
        if Sentence.aspect_label != -1:   # if the sentence has an aspect label,
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
    return review.num_stn_aspect_word,review.num_stn_aspect,review.num_stn_word,review.num_stn

    
class Bootstrapping:
    
    def sentence_label(self,corpus): # produce a label list
        if len(self.Aspect_Terms)>0:
            for company in corpus.Companies:
                for review in company.Reviews:
                    for Sentence in review.Sentences_info:
                        match_count=sent_aspect_match(Sentence,self.Aspect_Terms,len(self.Aspect_Terms))
                        if np.max(match_count)>0: # if at least one of the aspects has a match
                            s_label = np.where(np.max(match_count)==match_count)[0].tolist() # index of the aspect with max matched terms
                            Sentence.aspect_label = s_label ### *** TO DO: with tie
                        else:
                            pass
        else:
            print("Warning: No sentences or Aspect_Terms are recorded in this corpus")

    def calc_chi_sq(self,corpus):
        K = len(self.Aspect_Terms)
        V = len(corpus.Vocab)
        corpus.all_num_stn_aspect_word = np.zeros((K,V))
        corpus.all_num_stn_aspect = np.zeros(K)
        corpus.all_num_stn_word = np.zeros(V)
        corpus.all_num_stn = 0
        Chi_sq = np.zeros((K,V))
        if K>0:
            for company in corpus.Companies:
                for review in company.Reviews:
                    review.num_stn_aspect_word,review.num_stn_aspect,review.num_stn_word,review.num_stn = collect_stat_for_each_review(review,self.Aspect_Terms,corpus.Vocab)
                    corpus.all_num_stn += review.num_stn # total number of sentences with any aspect label
                    corpus.all_num_stn_aspect += review.num_stn_aspect
                    for w in review.UniWord:
                        z = np.where(w == review.UniWord)[0][0] # index, since the matrix for review is small
                        corpus.all_num_stn_word[w] += review.num_stn_word[z] # number of times aspect_i words (z) appear in all sentences
                        corpus.all_num_stn_aspect_word[:,w] += review.num_stn_aspect_word[:,z]

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

def load_Aspect_Terms(analyzer,filepath,VocabDict):
    '''
    # analyzer:    
    # filepath: path where the aspect seedwords text file is located
    # VocabDict: vocab lookup table
    '''    
    analyzer.Aspect_Terms=[]
    f = open(filepath, "r")
    for line in f:
        aspect = [VocabDict.get(stemmer.stem(w.strip().lower())) for w in line.split(",")]
        analyzer.Aspect_Terms.append(aspect)
    f.close()
    print("-------- Aspect Keywords loading completed!")

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
        t=0
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
            t=t+1
        print("complete iteration "+str(i+1)+"/"+str(NumIter))


def save_Aspect_Keywords_to_file(analyzer,filepath,Vocab):
    '''
    ###  INPUT
    # filepath: path where the complete aspect words text file will locate
    '''
    f = open(filepath, 'w')
    for aspect in analyzer.Aspect_Terms:
        for w in aspect:
            try:
                f.write(Vocab[w]+", ")
            except:
                pass
        f.write("\n\n\n")
    f.close()


def create_W_matrix_for_each_review(analyzer,review,corpus):
    Nd = len(review.UniWord)
    K=len(analyzer.Aspect_Terms)
    # V=len(corpus.Vocab)
    review.W = np.zeros((K,Nd))
    for k in range(K):
        for w in range(Nd):  ## w is index of UniWord_for_review
            # z = review.UniWord[w]
            if corpus.all_num_stn_aspect[k] > 0:
                review.W[k,w] = review.num_stn_aspect_word[k,w]/corpus.all_num_stn_aspect[k]


def create_all_W(analyzer,corpus):
    company_num=0
    for company in corpus.Companies:
        print("Creating W matrix for company '"+str(company_num)+"': "+company.Company)
        for review in company.Reviews:
            create_W_matrix_for_each_review(analyzer,review,corpus)
        company_num += 1


def produce_data_for_rating(analyzer,corpus,outputfolderpath,percompany=False):
    dir = outputfolderpath
    if not os.path.exists(dir):
        os.makedirs(dir)

    vocabfile = outputfolderpath / "vocab.txt"
    f = open(vocabfile,"w",encoding='UTF-8')
    for w in corpus.Vocab:
        f.write(w+",")
    f.close()
    
    vocabdictfile = outputfolderpath / "vocab_dict.txt"
    f = open(vocabdictfile,"w",encoding='UTF-8')
    with open(vocabdictfile,"w",encoding='utf-8') as f:
        json.dump(corpus.VocabDict, f)
    '''
    json.dumps(corpus.VocabDict)
    for w in corpus.VocabDict.items():
        f.write(w)
        f.write("\n")
    '''
    f.close()
    
    if percompany==False:
        reviewfile = outputfolderpath / "review_data_all.txt"
        f = open(reviewfile, 'w',encoding='UTF-8')
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
        reviewfile = outputfolderpath / "review_data.txt"
        f = open(reviewfile, 'w',encoding='UTF-8')
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