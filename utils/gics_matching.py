# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



### Only need to repeat once ###
path = '../sample_data/'
group = 'gind' # 'ggroup'

gics = pd.read_csv(path + 'master/S&P500_gics.csv')
gicstest = gics[pd.isnull(gics['indthru'])]

mapping = pd.read_csv(path + 'master/ticker_mapping.csv')
mapping['year'] = mapping['effthru'].apply(lambda x: int(x[:4]))
mappingtest = mapping[mapping['year']>=2019]

merged = pd.merge(gicstest, mappingtest, on='gvkey')
merged = merged[['gvkey', 'ggroup', 'gind', 'gsector', 'gsubind', 'conml']].drop_duplicates()
merged = merged.reset_index(drop=True)
merged['conml'] = merged['conml'].apply(lambda c: c.replace(' ','_'))
merged.rename(columns={'conml':'company'}, inplace=True)

origin = torch.load('../sample_data/master/review_metadata.pt')
origin = origin[['reviewId','company']]
sentence = torch.load('../sample_data/master/sentence_match.pt')    
final = pd.merge(sentence, origin, on='reviewId')[['textType', 'sentenceId', 'trigramSentence', 'company']]
del origin, sentence

sentence_gics = pd.merge(final, merged, on='company')
torch.save(sentence_gics, '../sample_data/master/sentence_gics.pt')

company_gics = sentence_gics[['company','gvkey','ggroup','gind','gsector','gsubind']].drop_duplicates()
company_gics = company_gics.reset_index(drop=True)
torch.save(company_gics, '../sample_data/master/company_gics.pt')
###


