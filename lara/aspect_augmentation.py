# -*- coding: utf-8 -*-


import os
import math
import torch
import string
import numpy as np
import pandas as pd
from nltk import FreqDist
from text_preprocessing import *
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
pd.set_option('display.max_columns', 500)
maketrans = ''.maketrans
replace_punctuation = maketrans(string.punctuation, ' '*len(string.punctuation))


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
        temp = [VocabDict[word] for word in sentence if word in VocabDict.keys()]
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
    def __init__(self, sentences, VocabDict, text_type):
        '''
        ###  INPUT
        # review_text: review text
        # VocabDict: vocabulary lookup table
        '''
        vocabdict_labeled_sentences = label_sentence_UseVocab(sentences, VocabDict)
        self.Sentences_info = [Sentence_info(sent) for sent in vocabdict_labeled_sentences]
        UniWord = {} # dictionary
        for sent in self.Sentences_info:
            UniWord = UniWord | sent.word_frequency.keys() # now, UniWord is a set
        #UniWord = {-1 if w == None else w for w in UniWord}
        self.UniWord = np.array([w for w in UniWord])
        self.UniWord.sort()
        self.NumOfUniWord = len(self.UniWord)


class Company:
    def __init__(self, master, company_name, VocabDict, text_type):
        '''
        ###  INPUT
        # company_name: company name ( e.g., 'XYZ International Inc.' )
        '''
        text_df =  master[master['company']==company_name][[t+'_tokenized_bigram_stemmed' for t in text_type]]
        stemmed_sentences = []
        for col in text_df.columns:
            stemmed_sentences += list(text_df[col])
        self.Reviews = [Review(sent, VocabDict, text_type) for sent in stemmed_sentences]
        self.NumOfReviews = len(self.Reviews)


class Corpus:
    def __init__(self, master, corpus, Vocab, VocabDict, text_type=['summary','pros','cons','advice']):
        '''
        ###  INPUT
        # corpus: list of all companies
        '''
        self.Vocab = Vocab
        self.Companies = [Company(master, company, VocabDict, text_type) for company in corpus]
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
    review.num_stn_aspect_word = np.zeros((K, review.NumOfUniWord))
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


def Add_Aspect_Keywords(analyzer, p, NumIter, c, K):
    '''
    ###  INPUT
    # analyzer
    # p: Maximum number of added aspect words in each round
    # NumIter: Maximum number of iterations
    # c: data
    '''
    for i in range(NumIter):
        analyzer.sentence_label(c, K)
        analyzer.calc_chi_sq(c, K)
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
                review.W[k, w] = review.num_stn_aspect_word[k, w] / corpus.all_num_stn_aspect[k]


def create_all_W(analyzer, corpus):
    company_num=0
    for company in corpus.Companies:
        print("Creating W matrix for company '"+str(company_num)+"': "+company.Company)
        for review in company.Reviews:
            create_W_matrix_for_each_review(analyzer, review, corpus)
        company_num += 1


def produce_data_for_rating(analyzer, corpus, output_path, percompany=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vocabfile = output_path + "vocab.txt"
    with open(vocabfile, 'w', encoding='utf8') as f:
        for w in corpus.Vocab:
            f.write(w+",")
    
    if percompany == False:
        reviewfile = output_path + "review_data_all.txt"
        with open(reviewfile, 'w', encoding='utf8') as f:
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
    else:
        reviewfile = output_path + "review_data.txt"
        with open(reviewfile, 'w', encoding='utf8') as f:
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
        
        
if __name__ == "__main__":
    
    # Call data
    path = '../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/'
    vocab = torch.load(path + 'english_glassdoor_reviews_english_vocab.pt')
    vocab_dict = torch.load(path + 'english_glassdoor_reviews_english_vocab_dict.pt')
    b_model =torch.load(path + 'english_glassdoor_reviews_english_bigram_model.pt')
    t_model = torch.load(path + 'english_glassdoor_reviews_english_trigram_model.pt')
    
    # Create analyzer
    analyzer = Bootstrapping()
    
    # Load aspect seedwords
    load_Aspect_Terms(analyzer, '../sample_data/lara/aspect_seed_words_bigrams_original.txt', vocab_dict)
    for aspect_num in analyzer.Aspect_Terms:
        print('-------- Aspect Seedwords:')
        print(aspect_num)
        print([vocab[num] for num in aspect_num])
    
    # Define corpus
    master = torch.load(+ 'english_glassdoor_reviews_text_preprocessed.pt')
    company_list = list(master['company'].unique())
    data = Corpus(master, company_list, vocab, vocab_dict)
    print('-------- Done creating corpus')
    
    # Labeling each sentence
    K = len(analyzer.Aspect_Terms)
    analyzer.sentence_label(data, K)
    print('-------- Done default labeling')
    
    # Calculate chi square
    analyzer.calc_chi_sq(data, K) # it works! CHECK LATER to see if +0.00001 is justified
    print('-------- Done with chi-square')
    
    # Update the aspect keywords list
    load_Aspect_Terms(analyzer, '../sample_data/lara/aspect_seed_words_bigrams_original.txt', vocab_dict)
    Add_Aspect_Keywords(analyzer, p=5, NumIter=7, c=data, K=K)
    
    # Check final results
    for aspect_num in analyzer.Aspect_Terms:
        print('-------- Final Aspect terms:')
        print(aspect_num)
        print([vocab[w] for w in aspect_num])
    
    
    # Save the aspect keywords
    aspectfile = path + 'aspect_final_words_bigrams_original.txt'
    f = open(aspectfile, 'w', encoding='utf8')
    
    for aspect in analyzer.Aspect_Terms:
        print('-------- Final Aspect terms:')
        for w in aspect:
            print(vocab[w])
            f.write(vocab[w])
            f.write(',')
        f.write('\n')
    f.close()

    """
    # Create W matrix for each review
    create_all_W(analyzer,data)
    
    # W matrix for all reviews
    produce_data_for_rating(analyzer, data, output_path, percompany=False)
    
    # W matrix for reviews per company
    produce_data_for_rating(analyzer, data, output_path, percompany=True)
    """
    
    