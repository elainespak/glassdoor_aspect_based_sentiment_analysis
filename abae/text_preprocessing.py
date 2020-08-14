# -*- coding: utf-8 -*-

import torch
import codecs
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', 500)


def make_train_test(all_tokenized_sentences, test=True, test_size=2000):
    if test==True:
        file = 'test'
        all_tokenized_sentences = all_tokenized_sentences[-test_size:]
    else:
        file = 'train'
        all_tokenized_sentences = all_tokenized_sentences[:-test_size]
    
    out = codecs.open('../sample_data/abae/glassdoor/'+file+'.txt', 'w', 'utf-8')
    for review in tqdm(all_tokenized_sentences):
        for tokens in review:
            if len(tokens) > 1:
                out.write(' '.join(tokens)+'\n')
                
    print('\nDone!')


if __name__ == "__main__":
    
    master = torch.load('../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/english_glassdoor_reviews_text_preprocessed.pt')
    
    all_tokenized_sentences = []
    for col in master:
        if col.endswith('_tokenized') and not col.startswith('summary'):
            all_tokenized_sentences += list(master[col])
    print('Done loading data!')
    
    make_train_test(all_tokenized_sentences, test=True, test_size=2000)