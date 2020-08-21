# -*- coding: utf-8 -*-

import os
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', 500)


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

def create_input(att_path, aspect, aspect_dictionary):
    
    with open(att_path, 'r') as f:
        att = f.read().split('----------------------------------------')
    with open(f'../abae/preprocessed_data/glassdoor/gold/{aspect}/{aspect}_000.txt', 'r') as f:
        gold = f.read().strip().split('\n')
    
    sent_dict = []
    for result in att[1:]:
        sent = result.strip().split('\n')
        temp = {'aspect_'+str(i): sent[i] for i in range(1,4)}
        temp['sentence'] = sent[4]
        temp['att'] = sent[5:]
        sent_dict.append(temp)
    
    df = pd.DataFrame(sent_dict)
    df['aspect'] = df['aspect_1'].apply(lambda a: d[a.split(':')[0]])
    #df['aspect'] = df.apply(lambda x: label_aspect(x['aspect_1'], x['aspect_2'], x['aspect_3'], d), axis=1)
    df[aspect] = gold
    aspect_df = df[df.aspect == aspect]
    return aspect_df
    

if __name__ == '__main__':
    
    # Parameters
    path = '../abae/output_dir/glassdoor/aspect_size_20/tests/'
    aspect = 'ratingCompensationAndBenefits'
    d = {'None': 'None',
         'Technical': '?', 'Restructuring': '?', 'Structure and Policies': '?', 'Customers, Products, and Services': '?', 'Workspace': '?', 'Location': '?',
         'Overall': 'Overall',
         'Benefits': 'CompensationAndBenefits', 'Perks': 'CompensationAndBenefits', 'Compensation': 'CompensationAndBenefits',
         'Moral Values': 'CultureAndValues', 'Culture': 'CultureAndValues', 'People': 'CultureAndValues',
         'Working Conditions': 'WorkLifeBalance', 'Work Life Balance': 'WorkLifeBalance',
         'Senior Leadership': 'SeniorLeadership',
         'Career Opportunities: Junior Perspective': 'CareerOpportunities', 'Career Opportunities: Senior Perspective': 'CareerOpportunities', 'Career Opportunities': 'CareerOpportunities'}
    
    # Make input data for sentence classification model
    data = pd.DataFrame()
    for att_file in tqdm(os.listdir(path)):
        path += att_file
        data = pd.concat([data, create_input(path, aspect, d)])
    data.to_csv(f'../sample_data/sentence_classification/{aspect}.csv')
    
    
    
path='../abae/output_dir/glassdoor/aspect_size_20/tests/att_weights_all_tokenized_trigram_sentences_000.txt.txt'
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


    
df['aspect'] = df['aspect_1'].apply(lambda a: d[a.split(':')[0]])
#df['aspect'] = df.apply(lambda x: label_aspect(x['aspect_1'], x['aspect_2'], x['aspect_3'], d), axis=1)

with open('../abae/preprocessed_data/glassdoor/gold/ratingCompensationAndBenefits/ratingCompensationAndBenefits_000.txt', 'r') as f:
    gold = f.read().strip().split('\n')
    
df['ratingCompensationAndBenefits'] = gold

test = df[df.aspect == 'CompensationAndBenefits']

len(test[test.ratingCompensationAndBenefits == '0.0']) / len(test)


