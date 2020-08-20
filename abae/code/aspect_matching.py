# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', 500)


path = 'C:/Users/elain/Desktop/glassdoor_aspect_based_sentiment_analysis/sample_data/'
master = torch.load(path + '2008 to 2018 SnP 500 Firm Data_Master English Files/english_glassdoor_reviews_text_preprocessed.pt')
print(master.head(10))

### Visualize % of missing ratings for each aspect
aspect_rating_count = len(master[master['ratingWorkLifeBalance']!=0.0])
all_count = len(master)
###

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




