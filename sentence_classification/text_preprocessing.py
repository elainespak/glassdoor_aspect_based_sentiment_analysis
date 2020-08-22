# -*- coding: utf-8 -*-

import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import sent_tokenize
pd.set_option('display.max_columns', 500)


# 1. Make exploded trigram data & rating data
path = '../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/'
master = torch.load(path + 'english_glassdoor_reviews_text_preprocessed.pt')
print(master.head(10))

master['all'] = master['pros_tokenized_trigram'] + master['cons_tokenized_trigram'] + master['advice_tokenized_trigram']
dat = master[['company', 'all'] + [col for col in master.columns if col.startswith('rating')]]
dat = dat.explode('all')
dat = dat[dat['all'].map(len) > 1]
dat.reset_index(inplace=True)
print(dat.head(10))

d_ceo = {'APPROVE': 3, 'NO_OPINION': 2, 'DISAPPROVE': 1, None: 0}
d_bus = {'POSITIVE': 3, 'NEUTRAL': 2, 'NEGATIVE': 1, None: 0}
d_friend = {'POSITIVE': 2, 'NEGATIVE': 1, None: 0}
dat['ratingCeo'] = dat['ratingCeo'].apply(lambda key: d_ceo[key])
dat['ratingBusinessOutlook'] = dat['ratingBusinessOutlook'].apply(lambda key: d_bus[key])
dat['ratingRecommendToFriend'] = dat['ratingRecommendToFriend'].apply(lambda key: d_friend[key])

# Save
torch.save(dat, '../output_dir/glassdoor/gold/english_review_exploded.pt')


def save_files(dat, i):
    i = '{0:0=3}'.format(i)
    for col in dat.columns:
        if col.startswith('rating'):
            np.savetxt(f'../output_dir/glassdoor/gold/{str(col)}/{str(col)}_{i}.txt', dat[col], fmt='%1.1f')
    
    with open(f'../output_dir/glassdoor/gold/sentences/all_tokenized_trigram_sentences_{i}.txt', 'w', encoding='utf8') as f:
        for sentence in tqdm(dat['all'].apply(lambda w: ' '.join(w))):
            f.write(sentence)
            f.write('\n')
    
n = 15000  #chunk row size
list_df = [dat[i:i+n] for i in range(0,dat.shape[0],n)]
for i, df in tqdm(enumerate(list_df)):
    save_files(df, i)


# 2. Make exploded data along with original sentences
def sentence_tokenize(raw):
    raw = re.sub('\r\n|\n-|\n|\r','. ', raw)
    raw = re.sub(',\.+ ', ', ', raw)
    raw = re.sub('\.+ ', '. ', raw)
    raw = re.sub('&amp;', '&', raw)
    
    sentences = sent_tokenize(raw)
    return sentences

path = '../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/'
master = torch.load(path + 'english_glassdoor_reviews_text_preprocessed.pt')
master['pros'] = master['pros'].apply(lambda x: sentence_tokenize(x))
print('done')
master['cons'] = master['cons'].apply(lambda x: sentence_tokenize(x))
print('done')
master['advice'] = master['advice'].apply(lambda x: sentence_tokenize(x))
master.head()
master['all'] = master['pros'] + master['cons'] + master['advice']
master['all_trigram'] = master['pros_tokenized_trigram'] + master['cons_tokenized_trigram'] + master['advice_tokenized_trigram']
master['allall'] = [tuple(x) for x in master[['all','all_trigram']].values.tolist()]

master['newall'] = ''
for idx in tqdm(range(len(master))):
    sent, tok = master['allall'][idx]
    keep = []    
    for j in range(len(tok)):
        if len(tok[j])>1:
            keep.append(sent[j])
    master['newall'][idx] = keep


dat = master.explode('newall')
dat2 = dat[~pd.isnull(dat['newall'])]
len(dat2)

ori = torch.load('../abae/preprocessed_data/glassdoor/gold/english_review_exploded.pt')
ori['original'] = list(dat2['newall'])

torch.save(ori, '../abae/preprocessed_data/glassdoor/gold/original_english_review_exploded.pt')

