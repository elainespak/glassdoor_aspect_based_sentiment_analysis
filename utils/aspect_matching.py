# -*- coding: utf-8 -*-

import os
import ast
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', 50)


def decode_pickle(dat):
    df = []
    d = {'words': '', 'sentence_embedding': '', 'unweighted_sentence_embedding': '', 'attention_weights': '',
         'aspect_1': '', 'aspect_2': '', 'aspect_3': '',
         'aspect_1_prob': '', 'aspect_2_prob': '', 'aspect_3_prob': ''
         }
    for item in dat:
        for key in item.keys():
            if type(item[key]) is bytes:
                temp = item[key].decode('utf-8')
            else:
                temp = item[key]
            d[key.decode('utf-8')] = temp
        df.append(d)
        d = {'words': '', 'sentence_embedding': '', 'unweighted_sentence_embedding': '', 'attention_weights': '',
         'aspect_1': '', 'aspect_2': '', 'aspect_3': '',
         'aspect_1_prob': '', 'aspect_2_prob': '', 'aspect_3_prob': ''
         }
    return df


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def label_aspect(a1, a2, a3, dictionary):
    a1 = dictionary[a1.split(':')[0]]
    a2 = dictionary[a2.split(':')[0]]
    a3 = dictionary[a3.split(':')[0]]
    if a1 in ['?', 'None'] and a2 in ['?', 'None']:
        return a3
    elif a1 in ['?', 'None']:
        return a2
    else:
        return a1


def create_input(att_path, aspect_dictionary):
    
    with open(att_path, 'r') as f:
        att = f.read().split('----------------------------------------')
    
    sent_dict = []
    for result in att[1:]:
        sent = result.strip().split('\n')
        temp = {'aspect_'+str(i): sent[i] for i in range(1,4)}
        temp['tokenized_sentence'] = sent[4]
        temp['att'] = sent[5:]
        sent_dict.append(temp)
    
    df = pd.DataFrame(sent_dict)
    df['aspect'] = df['aspect_1'].apply(lambda a: a.split(':')[0])
    #df['aspect'] = df.apply(lambda x: label_aspect(x['aspect_1'], x['aspect_2'], x['aspect_3'], d), axis=1)
    return df


def filter_by_aspect(data, aspect):
    data_aspect = data[data['aspect'] == aspect]
    data_aspect = data_aspect[['original', 'rating'+aspect]]
    data_aspect = data_aspect[data_aspect['rating'+aspect] != 0]
    data_aspect = data_aspect.reset_index(drop=True)
    data_aspect['rating'+aspect] = data_aspect['rating'+aspect].apply(lambda x: str(x-1))
    # index must begin with 0, otherwise triggers the device-side assert triggered error
    return data_aspect


if __name__ == '__main__':

    # Parameters
    texttype = 'cons'
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
    final = pd.merge(sentence, origin, on='reviewId')
    """
    print(df[20]['words'])
    print(df[20]['sentenceId'])
    print(df[400]['words'])
    print(df[400]['sentenceId'])
    print(df[3080010]['words'])
    print(df[3080010]['sentenceId'])
    
    origin[origin['sentenceId']==int(df[20]['sentenceId'])]
    origin[origin['sentenceId']==int(df[400]['sentenceId'])]
    origin[origin['sentenceId']==int(df[3080010]['sentenceId'])]
    """
    del origin, sentence
    
    df = pd.DataFrame(df)
    master = pd.merge(df, final[['sentenceId','trigramSentence','company']], on='sentenceId')
    del final, df
    


    master['aspect_1'].value_counts().plot.bar()
    company_list = list(master['company'].unique())
    aspect_list = list(master['aspect_1'].unique())
    
    all_sentence = {}
    for company in tqdm(company_list):
        for aspect in aspect_list:
            temp = master[(master['company']==company) & (master['aspect_1']==aspect)]
            if len(temp)==0:
                pass
            else:
                avg = np.mean(list(temp['sentence_embedding']), axis=0)
                all_sentence[(company, aspect)] = avg
    
    torch.save(all_sentence, path+'/aspect_size_12/average_sentence_embeddings.pt')
    
    microsoft = {}
    for (company, avg) in all_sentence.keys():
        try:
            microsoft[company] = cosine(all_sentence[('Microsoft_Corp','Leadership')],all_sentence[(company,'Leadership')])
        except:
            print(f'{company} had no reivew')
    
    result = [(k,v) for k, v in sorted(microsoft.items(), key=lambda item: item[1])]
    print(result[-40:])
    
    microsoft2 = {}
    for (company, avg) in all_sentence.keys():
        try:
            microsoft2[company] = cosine(all_sentence[('Microsoft_Corp','Work Hours')],all_sentence[(company,'Work Hours')])
        except:
            print(f'{company} had no reivew')
    
    result2 = [(k,v) for k, v in sorted(microsoft2.items(), key=lambda item: item[1])]
    print(result2[-40:])
    
    
