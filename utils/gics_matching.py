# -*- coding: utf-8 -*-

import torch
import pandas as pd
from aspect_matching import cosine
from tf_idf_keyword_extraction import calculate_tf_idf
pd.set_option('display.max_columns', 50)


path = '../sample_data/'
group = 'ggroup'
aspect = 'People and Culture'
company2 = 'Navient_Corp' # "Amazon.com_Inc" # McDonald's_Corp

gics = pd.read_csv(path + 'master/S&P500_gics.csv')
gicstest = gics[pd.isnull(gics['indthru'])]

mapping = pd.read_csv(path + 'master/ticker_mapping.csv')
mapping['year'] = mapping['effthru'].apply(lambda x: int(x[:4]))
mappingtest = mapping[mapping['year']>=2019]

merged = pd.merge(gicstest, mappingtest, on='gvkey')
final = merged.groupby(group)['conml'].apply(list).reset_index(name='conml')
final['conml'] = final['conml'].apply(lambda l: list(set(l)))
final = final.to_dict('records')

companies = final[15]['conml']
companies = [c.replace(' ', '_') for c in companies]

# Bring tf-idf
tfidf = torch.load(path+'abae/pros/aspect_size_12/master_tf_idf.pt')
tfidf = tfidf[tfidf['aspect_1']==aspect]
tfidf = calculate_tf_idf(tfidf)
tfidf['conml'] = tfidf.index

test = tfidf[tfidf['conml'].isin(companies)]
test['smart']

test2 = test[test['conml']==company2].transpose()
test2 = test2.drop(['conml'])
# Final results!
test2.sort_values(by=company2).tail(30)


# Cosine similarity
if __name__ == '__main__':
    all_sentence = torch.load(path+'abae/pros/aspect_size_12/average_sentence_embeddings.pt')
    
    microsoft = {}
    for (company, avg) in all_sentence.keys():
        if company in companies and avg == 'People and Culture':
            try:
                microsoft[company] = cosine(all_sentence[(company2,'People and Culture')],all_sentence[(company,avg)])
            except:
                print(f'{company} had no reivew')
    
    result = [(k,v) for k, v in sorted(microsoft.items(), key=lambda item: item[1])]
    print(result)
    