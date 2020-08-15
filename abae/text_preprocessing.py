# -*- coding: utf-8 -*-

import torch
import codecs
import argparse
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', 500)


def make_train_test(all_tokenized_sentences, outputpath, domain, test_size):
    train = all_tokenized_sentences[:-test_size]
    test = all_tokenized_sentences[-test_size:]
    
    out = codecs.open(outputpath+domain+'/train.txt', 'w', 'utf-8')
    for review in tqdm(train):
        for tokens in review:
            if len(tokens) > 1:
                out.write(' '.join(tokens)+'\n')
    out = codecs.open(outputpath+domain+'/test.txt', 'w', 'utf-8')
    for review in tqdm(test):
        for tokens in review:
            if len(tokens) > 1:
                out.write(' '.join(tokens)+'\n')
    print('\nDone!')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--masterpath', type=str, default='../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/english_glassdoor_reviews_text_preprocessed.pt')
    parser.add_argument('--outputpath', type=str, default='../preprocessed_data/')
    parser.add_argument('--domain', type=str)
    parser.add_argument('--tokentype', type=str)
    parser.add_argument('--testsize', type=int, default=2000)
    args = parser.parse_args()
    
    master = torch.load(args.masterpath)
    
    all_tokenized_sentences = []
    for col in master:
        if col.endswith(args.tokentype) and not col.startswith('summary'):
            all_tokenized_sentences += list(master[col])
    print('Done loading data!')
    
    make_train_test(all_tokenized_sentences, args.outputpath, args.domain, args.testsize)