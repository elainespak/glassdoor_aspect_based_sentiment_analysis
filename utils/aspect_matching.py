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

def fix_label(aspect_1, aspect_2, aspect_3):
    avoid = ['None']
    if aspect_1 in avoid and aspect_2 in avoid:
        return aspect_3
    elif aspect_1 in avoid:
        return aspect_2
    else:
        return aspect_1
        


if __name__ == '__main__':

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
    
    print(df['words'][400])
    print(sentence[sentence['sentenceId']==int(df['sentenceId'][400])]['trigramSentence'])
    print(df['words'][2080010])
    print(sentence[sentence['sentenceId']==int(df['sentenceId'][2080010])]['trigramSentence'])
    
    final = pd.merge(sentence, origin, on='reviewId')
    del origin, sentence
    master = pd.merge(df, final[['sentenceId','trigramSentence','company', 'reviewId']], on='sentenceId')
    del final, df
    
    if texttype == 'pros':
        master['aspect_1'] = master.apply(lambda df: fix_label(df['aspect_1'], df['aspect_2'],df['aspect_3']), axis=1)
        
        labels = {'People and Culture': 'CultureAndValues', 'Location': 'CultureAndValues', 'Overall': 'CultureAndValues',
                  'Pay': 'CompensationAndBenefits', 'Benefits': 'CompensationAndBenefits', 'Perks': 'CompensationAndBenefits',
                  'Career Opportunities': 'CareerOpportunities', 'Technology': 'CareerOpportunities',
                  'Work Life Balance': 'WorkLifeBalance',
                  'Leadership': 'SeniorLeadership',
                  'None': 'None',
                  'Company': 'BusinessOutlook', 
                  }
        master['aspect'] = master['aspect_1'].apply(lambda a: labels[a])
        print(master.head(10))
    elif texttype == 'cons':
        master['aspect_1'] = master.apply(lambda df: fix_label(df['aspect_1'], df['aspect_2'],df['aspect_3']), axis=1)
        
        labels = {'Leadership': 'SeniorLeadership',
                  'Overall - Negative': 'CultureAndValues', 'Culture': 'CultureAndValues', 'Overall': 'CultureAndValues', 'People': 'CultureAndValues',
                  'Pay': 'CompensationAndBenefits', 
                  'Technology': 'BusinessOutlook', 'Restructuring': 'BusinessOutlook', 'Company': 'BusinessOutlook',
                  'Structure': 'CareerOpportunities', 'Career Opportunities': 'CareerOpportunities',
                  'Work Hours': 'WorkLifeBalance'
                  }
        master['aspect'] = master['aspect_1'].apply(lambda a: labels[a])
        print(master.head(10))
    
    else:
        print(' --- None!')
    
    # Make evaluation dataframe
    torch.save(master[['sentenceId', 'reviewId', 'company', 'aspect', 'aspect_1', 'trigramSentence']],
               f'../sample_data/master/{texttype}_12_aspect_labeled.pt')
    
    
    # Make company embeddings
    master['aspect_1'].value_counts().plot.bar()
    master['aspect'].value_counts().plot.bar()
    company_list = list(master['company'].unique())
    aspect_list = list(master['aspect_1'].unique())
    
    all_sentence = {}
    all_sentence_unweighted = {}
    for company in tqdm(company_list):
        for aspect in aspect_list:
            temp = master[(master['company']==company) & (master['aspect_1']==aspect)]
            if len(temp)==0:
                pass
            else:
                avg = np.mean(list(temp['sentence_embedding']), axis=0)
                all_sentence[(company, aspect)] = avg
                unweighted_avg = np.mean(list(temp['unweighted_sentence_embedding']), axis=0)
                all_sentence_unweighted[(company, aspect)] = unweighted_avg
    
    torch.save(all_sentence, path+'/aspect_size_12/average_sentence_embeddings.pt')
    torch.save(all_sentence_unweighted, path+'/aspect_size_12/unweighted_average_sentence_embeddings.pt')
    

    # Done
    del master