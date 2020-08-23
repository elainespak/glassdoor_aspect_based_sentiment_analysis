# -*- coding: utf-8 -*-

import os
import torch
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
    df['aspect'] = df['aspect_1'].apply(lambda a: d[a.split(':')[0]])
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
    path = '../abae/output_dir/glassdoor/aspect_size_20/tests/'
    d = {# None
         'None': 'None',
         # ?
         'Technical': '?', 'Restructuring': '?', 'Structure and Policies': '?',
         'Customers, Products, and Services': '?', 'Workspace': '?', 'Location': '?',
         # Overall
         'Overall': 'Overall',
         # CompensationAndBenefits
         'Benefits': 'CompensationAndBenefits', 'Perks': 'CompensationAndBenefits', 'Compensation': 'CompensationAndBenefits',
         # CultureAndValues
         'Moral Values': 'CultureAndValues', 'Culture': 'CultureAndValues', 'People': 'CultureAndValues',
         # WorkLifeBalance
         'Working Conditions': 'WorkLifeBalance', 'Work Life Balance': 'WorkLifeBalance',
         #SeniorLeadership
         'Senior Leadership': 'SeniorLeadership',
         # CareerOpportunities
         'Career Opportunities: Junior Perspective': 'CareerOpportunities',
         'Career Opportunities: Senior Perspective': 'CareerOpportunities',
         'Career Opportunities': 'CareerOpportunities'}
    aspects = {'Overall', 'CompensationAndBenefits', 'CultureAndValues', 'WorkLifeBalance', 'SeniorLeadership', 'CareerOpportunities'}
    
    # Make input data for sentence classification model
    label = pd.DataFrame()
    for att_file in tqdm(os.listdir(path)):
        filepath = path + att_file
        temp = create_input(filepath, d)[['tokenized_sentence', 'aspect']]
        label = pd.concat([label, temp])
    label = label.reset_index(drop=True)
    
    origin = torch.load('../abae/preprocessed_data/glassdoor/gold/original_english_review_exploded.pt')
    origin = origin[['original']+[col for col in origin.columns if col.startswith('rating')]]
    
    data = pd.concat([origin, label], axis=1)
    
    for aspect in aspects:
        data_aspect = filter_by_aspect(data, aspect)
        torch.save(data_aspect, f'../sample_data/sentence_classification/{aspect}_for_sentence_classification.pt')
        print(f'Done with {aspect}')
       
