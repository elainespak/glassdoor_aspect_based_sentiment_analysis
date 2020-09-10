# -*- coding: utf-8 -*-

import torch
import codecs
import argparse
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', 50)


def save_to_textfile(out, sentences):
    for tokens in tqdm(sentences):
        out.write(' '.join(tokens)+'\n')
    
    
def make_train_test(df, task, outputpath, sentence_type, test_size):
    
    df = df.sample(frac = 1)
    sentences = df[sentence_type]
    indices = df['sentenceId']
    
    if task == 'master':
        out = codecs.open(outputpath+'train.txt', 'w', 'utf-8')
        save_to_textfile(out, sentences)
        out = codecs.open(outputpath+'test.txt', 'w', 'utf-8')
        save_to_textfile(out, sentences)
        out = codecs.open(outputpath+'indices.txt', 'w', 'utf-8')
        for idx in indices:
            out.write(str(idx)+'\n')
    else:
        train, test = sentences[:-test_size], sentences[-test_size:]
        train_indices, test_indices = indices[:-test_size], indices[-test_size:]
        
        out = codecs.open(outputpath+'train.txt', 'w', 'utf-8')
        save_to_textfile(out, train)
        out = codecs.open(outputpath+'test.txt', 'w', 'utf-8')
        save_to_textfile(out, test)
        out = codecs.open(outputpath+'train_indices.txt', 'w', 'utf-8')
        for idx in train_indices:
            out.write(str(idx)+'\n')
        out = codecs.open(outputpath+'test_indices.txt', 'w', 'utf-8')
        for idx in test_indices:
            out.write(str(idx)+'\n')
    
    print('\nDone!')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--masterpath', type=str, default='../sample_data/master/sentence_match.pt')
    parser.add_argument('--outputpath', type=str, default='../sample_data/abae/')
    parser.add_argument('--sentencetype', type=str, default='trigramSentence')
    parser.add_argument('--task', type=str, default='master')
    parser.add_argument('--testsize', type=int, default=2000)
    args = parser.parse_args()
    
    match = torch.load(args.masterpath)
    print('Done loading original data!')
    make_train_test(match, args.task, args.outputpath, args.sentencetype, args.testsize)