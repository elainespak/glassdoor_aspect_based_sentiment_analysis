# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:33:16 2019

@author: elain
"""

from pathlib import Path
import numpy as np
import os
import math
import re
import sys
import json
import string
import itertools
try:
    maketrans = ''.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import gensim
import langid


####### ---------------------------- parameters ---------------------------- ####### 
stemmer = PorterStemmer()

os.chdir(r"C:\Users\elain\Desktop\SnP 500 Analysis\LARA-Python")
p = Path(os.getcwd())
#folderpath = p.parent / '2018 SnP 500 Firm Data All'
inputfolderpath = p / 'glassdoor_mini_input'
outputfolderpath = p / 'glassdoor_mini_output_alltext' / 'aspect_segmentation'

all_reviews = []
company_list = []
remove_list = ['Abbott', 'Abbott Laboratories',
               'Chevron', 'Netflix', 'Eaton', 'Electronic Arts',
               'General Motors', 'GM', 'Merril Lynch', 'Merrill Lynch', 'NextEra', 'NextEra Energy',
               '3M']
remove_list = [r.lower() for r in remove_list]

#text_type = 'pros'
text_type = ['summary', 'pros', 'cons', 'advice']




####### ---------------------------- preprocess.py ---------------------------- ####### 

# load all review texts at once
def load_file(file, text='pros'):    
    f = json.load(open(file)) # load each company's review text json file
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
                else:
                    pass # Not all parts are in English, so discard the review
            except:
                pass
                #print('error with langdetect')
    
    reviews = [t[0] for t in temp]
    ratings = [t[1] for t in temp]

    return reviews, ratings


def parse_all_reviews_to_sentence(reviews, remove_company_list):
    review_processed = []
    actual = []

    only_sent = []
    for r in reviews:
        sentences = nltk.sent_tokenize(r)
        if len(sentences) < 2:
            sentences = re.split('\r\n|\n-',r) # Often, glassdoor reviews are in a list
        if len(sentences) < 2:
            sentences = re.split('\r|\n',r) # Often, glassdoor reviews are in a list
        actual.append(sentences)
        sent = []
        for s in sentences:
			#words to lower case
            s = s.lower()
            #remove company mentions in the sentences
            for company_mention in remove_company_list:
                if company_mention in s:
                    s = s.replace(company_mention,'')
			#remove punctuations and stopwords
            replace_punctuation = maketrans(string.punctuation, ' '*len(string.punctuation))
            s = s.translate(replace_punctuation)
            stop_words = list(stopwords.words('english'))
            additional_stopwords = ["'s","...","'ve","``","''","'m",'--',"'ll","'d"]
			# additional_stopwords = []
            stop_words = set(stop_words + additional_stopwords)
			# sys.exit()
            word_tokens = word_tokenize(s)
            s = [w for w in word_tokens if not w in stop_words]
            sent.append(s)
        review_processed.append(sent)
            #Porter Stemmer
            #stemmed = [stemmer.stem(w) for w in s]
            #if len(stemmed)>0:
            #    sent.append(stemmed)
        only_sent.extend(sent)
    return review_processed, actual, only_sent # only_sent is each review split into sentences


def make_ngrams_model(tokenized_sents, set_min_count=5, set_threshold=60):
    bigram = gensim.models.Phrases(tokenized_sents, min_count=set_min_count, threshold=set_threshold)
    trigram = gensim.models.Phrases(bigram[tokenized_sents], threshold=set_threshold)
    
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    return bigram_mod, trigram_mod


def stemming(tokenized_sents):
    sentences = []
    
    for sent in tokenized_sents:
        stemmed = [stemmer.stem(w) for w in sent]
        if len(stemmed)>0:
            sentences.append(stemmed)
        
    return sentences


def To_One_List(lists):
    '''
    # list of lists to one list , e.g. [[1,2],[3,4]] -> [1,2,3,4]
    # from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    '''
    return list(itertools.chain.from_iterable(lists))

def create_vocab(sent):
    words = To_One_List(sent) # Gather every word from every sentence into one list
    freq = FreqDist(words) # Count occurrence of every word
    vocab = [k for k,v in freq.items() if v > 0] # Create the official "vocab" with only frequent words
    vocab_dict = dict(zip(vocab, range(len(vocab)))) # Assign a special unique number corresponding to each word
    return vocab, vocab_dict

####### ---------------------------- preprocess.py ---------------------------- ####### 
    


####### ---------------------------- aspect_augmentation.py ---------------------------- ####### 

def make_ngrams(bigram_mod, trigram_mod, tokenized_sents):
    b = [bigram_mod[sent] for sent in tokenized_sents]
    t = [trigram_mod[bigram_mod[sent]] for sent in tokenized_sents]
    return b, t

def parse_to_sentence(review):
    '''
    # INPUT
    # review: a single review data
              e.g., '\n-Employee recognition\n-We are definitely a team'
    
    # OUTPUT
    # only_sent: a list of individual sentences with stemmed/lemmatized words and no stopwords
                 e.g., [['employe', 'recognit'],['definit', 'team']]
    '''
    only_sent = []
    sentences = nltk.sent_tokenize(review)
    if len(sentences) < 2:
        sentences = re.split('\r\n|\n-|\n',review) # Often, glassdoor reviews are in a list
    if len(sentences) < 2:
        sentences = re.split('\r|\n',review)
    sent = []
    for s in sentences:
		#words to lower case
        s = s.lower()
        #remove company mentions in the sentences
        for company_mention in remove_list:
            if company_mention in s:
                s = s.replace(company_mention,'')
		#remove punctuations and stopwords
        replace_punctuation = maketrans(string.punctuation, ' '*len(string.punctuation))
        s = s.translate(replace_punctuation)
        stop_words = list(stopwords.words('english'))
        additional_stopwords = ["'s","...","'ve","``","''","'m",'--',"'ll","'d"]
		# additional_stopwords = []
        stop_words = set(stop_words + additional_stopwords)
		# print stop_words
		# sys.exit()
        word_tokens = word_tokenize(s)
        s = [w for w in word_tokens if not w in stop_words]
        sent.append(s)
        #Porter Stemmer
        #stemmed = [stemmer.stem(w) for w in s]
        #if len(stemmed)>0:
        #    sent.append(stemmed)
    only_sent.extend(sent)
    return only_sent

def label_sentence_UseVocab(only_sentences,VocabDict):
    '''
    Label every word of a sentence by using:
    1) a corresponding number from the VocabDict (vocabulary lookup table)
        OR
    2) "None" label
    
    '''
    num_sent = []
    for sent in only_sentences:
        temp = [VocabDict.get(w) for w in sent]
        #temp = [-1 if w == None else w for w in temp]
        if len(temp)>0:
            num_sent.append(temp)
    return num_sent


class Sentence_info:
    def __init__(self, sent):
        '''
        # INPUT
        # sent: num_sent (e.g., '-Amazing people' --> [0, 1]) ###['amaz', 'peopl'])
        
        # DEFINED
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
        # INPUT
        # review_data: each individual review
        # VocabDict: vocabulary lookup table
        
        # DEFINED
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
        '''
        UniWord = {}
        for sent in self.Sentences_info:
            UniWord = UniWord | sent.word_frequency.keys()
        UniWord = {w for w in UniWord if w != None} # CHECK LATER
        # UniWord = {-1 if w == None else w for w in UniWord}
        self.UniWord = np.array([w for w in UniWord])
        self.UniWord.sort()
        self.NumOfUniWord = len(self.UniWord) # 2 for 'this restaurant is not worth it.' sentence
        '''
        
def load_review_file(company_name):
    filename = company_name.replace(' ', '_')+'_individual_reviews_all.txt'
    file_name = inputfolderpath / filename
    reviews = json.load(open(file_name)).values()
    final_reviews = []
    
    if type(text_type) != list:
        for r in reviews:
            if langid.classify(r.get(text_type))[0]=='en':
                final_reviews.append(r)
    else:
        for r in reviews:
            try:
                is_all_english = [langid.classify(r.get(t))[0] for t in text_type]
                if list(set(is_all_english)) == ['en']:
                    final_reviews.append(r)
                else:
                    pass # Not all review parts are in English
            except:
                pass
                
    return final_reviews

class Company:
    def __init__(self, company_name, VocabDict):
        '''
        # INPUT
        # company_name: company name
                        e.g., 'XYZ International Inc.'
        # company_data: json file itself
        '''
        #self.Company = re.match('.*?(?=_individual_reviews_all.txt)', company_json)
        #self.Reviews = [Review(review, VocabDict) for review in company_data.get("Reviews")]
        self.Company = company_name
        self.Reviews = [Review(review, VocabDict) for review in load_review_file(self.Company)]
        self.NumOfReviews = len(self.Reviews)
    
def compare_label(label, l):
    '''
    ????? NOT SURE WHAT THIS FUNCTION DOES..
    '''
    return label in l

def sent_aspect_match(Sentences_info, aspects, K):
    '''
    # INPUT
    # aspects: list of list of aspects
               e.g., [['pay','money','benefits'], ['coworkers','team','colleagues']]
    # k: number of different aspects
    
    # OUTPUT
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
        # INPUT
        # corpus: list of all companies
        # Vocab: all Vocab??????????
        # Count: 
        '''
        self.Vocab = Vocab
        self.VocabDict = VocabDict
        #self.VocabTF = Count
        self.V = len(Vocab)
        self.Aspect_Terms = []
        self.Companies = [Company(company_name, self.VocabDict) for company_name in corpus] # rest = one restaurant
        self.NumOfCompanies = len(corpus)


def ChisqTest(N, taDF, tDF, aDF):
    '''
    # INPUT
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
    # INPUT
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
    def sentence_label(self,corpus):   # produce a label list
        if len(self.Aspect_Terms)>0:
            keep_match_count = 0 # DELETE LATER
            for company in corpus.Companies:
                for review in company.Reviews:
                    keep_i = 0 # DELETE LATER
                    for Sentence in review.Sentences_info:
                        match_count=sent_aspect_match(Sentence,self.Aspect_Terms,len(self.Aspect_Terms))
                        if np.max(match_count)>0: # if at least one of the aspects has a match
                            s_label = np.where(np.max(match_count)==match_count)[0].tolist() # index of the aspect with max matched terms
                            Sentence.aspect_label = s_label ### *** TO DO: with tie
                            #print(str(review.only_sents[keep_i]) + ": Match") # The sentence has a match! # DELETE LATER
                        else:
                            pass # The sentence has no match with any of the aspects!
                            #print(str(review.only_sents[keep_i])) # DELETE LATER
                            keep_match_count += 1 # DELETE LATER
                        keep_i += 1 # DELETE LATER
            print('-------- Sentences with no match:\n'+ str(keep_match_count))
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
    # INPUT
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
                if w not in To_One_List(analyzer.Aspect_Terms):
                    analyzer.Aspect_Terms[t].append(w)
                    aspect_num += 1
                if aspect_num > p:
                    break
            t=t+1
        print("complete iteration "+str(i+1)+"/"+str(NumIter))


def save_Aspect_Keywords_to_file(analyzer,filepath,Vocab):
    '''
    # INPUT
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
    
    
####### ---------------------------- aspect_augmentation.py ---------------------------- ####### 



   
####### ---------------------------- run.py ---------------------------- ####### 
    
##### ========================= #####
#####  1. Create VocabDict
##### ========================= #####

for file in os.listdir(inputfolderpath):
    if file.endswith('_individual_reviews_all.txt'):
        temp_reviews, temp_ratings = load_file(inputfolderpath / file, text=text_type)
        print(f'{file} successfully loaded')
        company_name = re.match('.*(?=_individual_reviews_all.txt)',file).group(0)
        company_name = company_name.replace('_', ' ')
        company_list.append(company_name)
        for r in temp_reviews:
            all_reviews.append(r)

# This takes long!! Almost 1 hour to process 500 companies
all_review_processed, all_actual, all_only_sent = parse_all_reviews_to_sentence(all_reviews, remove_list)

# Create bigram and trigram models
b_mod, t_mod = make_ngrams_model(all_only_sent, 5, 20) # for now , just use bigram

# Process sentences with b_mod
all_only_sent, t = make_ngrams(bigram_mod=b_mod, trigram_mod=t_mod, tokenized_sents = all_only_sent)

# Stemming
only_sent = stemming(all_only_sent)

# Create vocabs list and vocabs dictionary
vocab, vocab_dict = create_vocab(only_sent)


### ------------------------------------ Check how frequent the bigrams are!
test = To_One_List(only_sent)
test_freq = FreqDist(test)
ok = sorted([(test_freq[k],k) for k,v in vocab_dict.items() if '_' in k], reverse=True)
print(ok[:15])
print(ok[15:30])
### ------------------------------------------------------------------------


# Create analyzer
analyzer = Bootstrapping()

# Load aspect seedwords
load_Aspect_Terms(analyzer, inputfolderpath/'aspect_seed_words_bigrams.txt', vocab_dict)
for aspect in analyzer.Aspect_Terms:
    print('-------- Aspect Seedwords:')
    print(aspect)
    print([vocab[w] for w in aspect])

# Define corpus (test with two companies)
data = Corpus(company_list, vocab, vocab_dict)

# Labeling each sentence
analyzer.sentence_label(data)

# Calculate chi square
analyzer.calc_chi_sq(data) # it works! CHECK LATER to see if +0.00001 is justified

# Update the aspect keywords list
load_Aspect_Terms(analyzer, inputfolderpath/'aspect_seed_words_bigrams.txt', vocab_dict)
Add_Aspect_Keywords(analyzer, p=5, NumIter=5, c=data)

# Save the aspect keywords
aspectfile = outputfolderpath / "aspect_final_words_bigrams_8.txt"
f = open(aspectfile, 'w',encoding='UTF-8')

for aspect in analyzer.Aspect_Terms:
    print('-------- Final Aspect terms:')
    for w in aspect:
        print(vocab[w])
        f.write(vocab[w])
        f.write(',')
    f.write('\n')
f.close()

# Create W matrix for each review
create_all_W(analyzer,data)

# W matrix for all reviews
produce_data_for_rating(analyzer,data,outputfolderpath,percompany=False)

# W matrix for reviews per company
produce_data_for_rating(analyzer,data,outputfolderpath,percompany=True)


