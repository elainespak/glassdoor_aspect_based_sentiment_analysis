# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', 500)


d = {'None': 'None', 'Culture': 'CultureAndValues', 'Perks': 'CompensationAndBenefits',
'Technical': '?', 'Overall': 'Overall', 'Benefits': 'CompensationAndBenefits', 'None': 'None',
'Restructuring': '?', 'Structure and Policies': '?', 'Customers, Products, and Services': '?',
'Moral Values': 'CultureAndValues', 'Workspace': '?', 'Working Conditions': 'WorkLifeBalance',
'Senior Leadership': 'SeniorLeadership', 'Work Life Balance': 'WorkLifeBalance',
'Location': '?', 'Career Opportunities: Junior Perspective': 'CareerOpportunities',
'Compensation': 'CompensationAndBenefits',
'Career Opportunities: Senior Perspective': 'CareerOpportunities', 'People': 'CultureAndValues',
'Career Opportunities': 'CareerOpportunities'}

path='../output_dir/glassdoor/aspect_size_20/tests/att_weights_all_tokenized_trigram_sentences_000.txt.txt'
with open(path, 'r') as f:
    test = f.read().split('----------------------------------------')

sent_dict = []
for result in test[1:]:
    sent = result.strip().split('\n')
    temp = {'aspect_'+str(i): sent[i] for i in range(1,4)}
    temp['sentence'] = sent[4]
    temp['att'] = sent[5:]
    sent_dict.append(temp)

df = pd.DataFrame(sent_dict)
len(df[df.ratingCompensationAndBenefits == '0.0']) / len(df)

def aspect(a1, a2, a3, dictionary):
    a1 = dictionary[a1.split(':')[0]]
    a2 = dictionary[a2.split(':')[0]]
    a3 = dictionary[a3.split(':')[0]]
    if a1 in ['?', 'None'] and a2 in ['?', 'None']:
        return a3
    elif a1 in ['?', 'None']:
        return a2
    else:
        return a1
    
df['aspect'] = df['aspect_1'].apply(lambda a: d[a.split(':')[0]])
#df['aspect'] = df.apply(lambda x: aspect(x['aspect_1'], x['aspect_2'], x['aspect_3'], d), axis=1)

with open('../preprocessed_data/glassdoor/gold/ratingCompensationAndBenefits/ratingCompensationAndBenefits_000.txt', 'r') as f:
    gold = f.read().strip().split('\n')
    
df['ratingCompensationAndBenefits'] = gold

test = df[df.aspect == 'CompensationAndBenefits']

len(test[test.ratingCompensationAndBenefits == '0.0']) / len(test)


