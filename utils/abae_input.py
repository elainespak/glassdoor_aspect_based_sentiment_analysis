# -*- coding: utf-8 -*-

import os
import torch
import codecs
import argparse
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', 50)


def make_folder(outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

def chunks(l, n):
    result = [l[i:i+n] for i in range(0, len(l), n)]
    return result
    
def save_to_textfile(out, sentences):
    for tokens in tqdm(sentences):
        out.write(' '.join(tokens)+'\n')

def save_split_tests(outputpath, test):
    outputpath = outputpath + 'tests/'
    make_folder(outputpath)
    tests = chunks(test, 15000)
    for i in tqdm(range(len(tests))):
        idx = '{0:0=3d}'.format(i+1)
        out = codecs.open(f'{outputpath}test_{idx}.txt', 'w', 'utf-8')
        save_to_textfile(out, tests[i])    
    
def make_train_test(df, outputpath, task, text_type, sentence_type, test_size):
    if text_type == 'all':
        pass
    else:
        df = df[df['textType']==text_type]
    df = df.sample(frac = 1)
    sentences = df[sentence_type]
    indices = df['sentenceId']
    
    if task == 'master':
        out = codecs.open(outputpath+'train.txt', 'w', 'utf-8')
        save_to_textfile(out, sentences)
        out = codecs.open(outputpath+'test.txt', 'w', 'utf-8')
        save_to_textfile(out, sentences)
        save_split_tests(outputpath, sentences)
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
        save_split_tests(outputpath, test)
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
    parser.add_argument('--texttype',type=str,default='all')
    args = parser.parse_args()
    
    print(f'Domain is {args.texttype}')
    match = torch.load(args.masterpath)
    print('Done loading original data!')

    outputpath = args.outputpath + args.texttype+'/'
    make_folder(outputpath)
    
    make_train_test(match, outputpath, args.task, args.texttype, args.sentencetype, args.testsize)