# -*- coding: utf-8 -*-

# Import necessary packages
import re
import json
import pickle
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models.phrases import Phrases, Phraser

def get_sentences(input_file_pointer):
    while True:
        line = input_file_pointer.readline()
        if not line:
            break
        yield line
        
def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    return re.sub(r'\s{2,}', ' ', sentence)

def tokenize(sentence, replace_punctuation):
    sentence = sentence.translate(replace_punctuation)
    return [token for token in sentence.split() if token not in stopwords]#STOP_WORDS]

def build_phrases(sentences):
    phrases = Phrases(sentences,
                      min_count=5,
                      threshold=7,
                      progress_per=1000)
    return Phraser(phrases)


if __name__ == "__main__":
    
    # Bring in the file
    alldat = pd.read_csv('./sample_data/2008_to_2018_SnP500_Names.csv', delimiter=',')
    names = list(alldat['conml'])
    gvkeys = list(alldat['gvkey'])
    companies_all = [(names[i],gvkeys[i]) for i in range(len(alldat))]
    print(f'company-gvkey pair check: \n{companies_all[:3]}')
    
    # 1. Build 1-gram sentences
    sentences = []
    replace_punctuation = string.maketrans(string.punctuation,
                                                   ' '*len(string.punctuation))
    for name,gvkey in tqdm(companies_all):
        name = name.replace(' ','_')
        try:
            jsondat = json.load(open('./sample_data/2008 to 2018 SnP 500 Firm Data All/'+name+'_individual_reviews_all.txt'))
            for v in list(jsondat.values()):
                cleaned_sentence = clean_sentence(v['pros'])
                tokenized_sentence = tokenize(cleaned_sentence)
                sentences.append(tokenized_sentence)
                
                cleaned_sentence = clean_sentence(v['cons'])
                tokenized_sentence = tokenize(cleaned_sentence)
                sentences.append(tokenized_sentence)
        except:
            print(f'No glassdoor data for {name}')
    print(f'total length of sentences: \n{len(sentences)}')
    # Save 1-gram sentences
    with open('./sample_output/preprocessed_sentences_1gram.pkl', 'wb') as f:
        pickle.dump(sentences, f)
    
    # 2. Build 2-gram sentences
    sentences2_model = build_phrases(sentences)
    sentences2 = list(sentences2_model[sentences]) # Add bi-grams to the corpus
    print(f'example of bi-gram sentence: \n{sentences2[9]}')
    # Save 2-gram sentences
    with open('./sample_output/preprocessed_sentences_2gram.pkl', 'wb') as f:
        pickle.dump(sentences, f)