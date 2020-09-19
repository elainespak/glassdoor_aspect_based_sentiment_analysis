# -*- coding: utf-8 -*-

import os
import ast
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from chi_square import to_one_list
from aspect_matching import decode_pickle
pd.set_option('display.max_columns', 50)


def calculate_tf_idf(df):
    vectorizer = TfidfVectorizer()
    words = [' '.join(w) for w in df['trigramSentence']]
    vectors = vectorizer.fit_transform(words)
    #print('Done vectorizing for tf-idf')
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    result = pd.DataFrame(denselist, columns=feature_names)
    result.index = list(df['company'])
    return result

def get_most_common_trigrams(words, n=100):
    keywords = Counter(w for w in words if len(w.split('_')) >= 3).most_common(n)
    return [word for word, freq in keywords]


if __name__ == "__main__":

    # Parameters
    texttype = 'pros'
    path = '../sample_data/abae/'+texttype
    
    # Bring data
    with open(path + '/aspect_size_12/cluster_map.txt', 'r') as f:
        cluster_map = f.readlines()
    cluster_map = ''.join([i.replace('\n', '') for i in cluster_map])
    cluster_map = ast.literal_eval(cluster_map)
    
    testpath = path + '/aspect_size_12/tests_results/'
    df = []
    for file in tqdm(os.listdir(testpath)):
        with open(testpath+file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        df += decode_pickle(data)
    
    with open(path + '/indices.txt', 'r') as f:
        indices = f.readlines()
    indices = [i.replace('\n','') for i in indices]
    
    for d, idx in zip(df, indices):
        d['sentenceId'] = int(idx)
    
    # Match sentence with company
    origin = torch.load('../sample_data/master/review_metadata.pt')
    origin = origin[['reviewId','company']]
    sentence = torch.load('../sample_data/master/sentence_match.pt')
    df = pd.DataFrame(df)
    final = pd.merge(sentence, origin, on='reviewId')
    del origin, sentence
    
    master = pd.merge(df, final[['sentenceId','trigramSentence','company']], on='sentenceId')
    del final, df
    
    # tf-idf
    master2 = master.groupby(['company','aspect_1'])['trigramSentence'].apply(list).reset_index(name='trigramSentence')
    del master
    master2['trigramSentence'] = master2['trigramSentence'].apply(lambda s: to_one_list(s))
    torch.save(master2, path+'/aspect_size_12/master_tf_idf.pt')
    
    # for each aspect...
    test = master2[master2['aspect_1']=='People and Culture']
    tfidf = calculate_tf_idf(test)
    
    