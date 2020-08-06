# -*- coding: utf-8 -*-

import torch
import codecs
import pandas as pd
from tqdm import tqdm

pd.set_option('display.max_columns', 500)



def parseSentence(line):
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem

def preprocess_train(domain):
    f = codecs.open('../datasets/'+domain+'/train.txt', 'r', 'utf-8')
    out = codecs.open('../preprocessed_data/'+domain+'/train.txt', 'w', 'utf-8')

    for line in f:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out.write(' '.join(tokens)+'\n')

def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open('../datasets/'+domain+'/test.txt', 'r', 'utf-8')
    f2 = codecs.open('../datasets/'+domain+'/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open('../preprocessed_data/'+domain+'/test.txt', 'w', 'utf-8')
    out2 = codecs.open('../preprocessed_data/'+domain+'/test_label.txt', 'w', 'utf-8')

    for text, label in zip(f1, f2):
        label = label.strip()
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
            continue
        tokens = parseSentence(text)
        if len(tokens) > 0:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label+'\n')

def preprocess(domain):
    print '\t'+domain+' train set ...'
    preprocess_train(domain)
    print '\t'+domain+' test set ...'
    preprocess_test(domain)
    
    
if __name__ == "__main__":
    master = torch.load('../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/english_glassdoor_reviews_text_preprocessed.pt')
    
    all_tokenized_sentences = []
    for col in master:
        if col.endswith('_tokenized'):
            all_tokenized_sentences += list(master[col])
    
    domain = 'glassdoor'
    out = codecs.open('../sample_data/abae/'+domain+'/train.txt', 'w', 'utf-8')
    for review in tqdm(all_tokenized_sentences):
        for tokens in review:
            if len(tokens) > 1:
                out.write(' '.join(tokens)+'\n')