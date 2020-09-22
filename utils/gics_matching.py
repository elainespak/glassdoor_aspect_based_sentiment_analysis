# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from tf_idf_keyword_extraction import calculate_tf_idf
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

final = pd.merge(final, merged, on='company')
torch.save(final, '../sample_data/master/sentence_gics.pt')
###


# Cosine similarity
if __name__ == '__main__':
    
    # Parameters
    path = '../sample_data/'
    group = 'gind' # 'ggroup'
    text_type = 'pros'
    df = torch.load(path+'master/sentence_gics.pt')
    
    groups = list(df[group].unique())
    companies = list(df[df[group]==452020]['company'].unique())
    company_of_interest = 'Seagate_Technology_Plc'
    aspect_of_interest = 'Benefits'
    
    # ABAE sentence vector averaged
    all_sentence = torch.load(path+f'abae/{text_type}/aspect_size_12/unweighted_average_sentence_embeddings.pt')
    
    aspects = list(set([k[1] for k in all_sentence.keys()]))
    mine = {}
    for (company, avg) in all_sentence.keys():
        if company in companies and avg == aspect_of_interest:
            try:
                mine[company] = cosine(all_sentence[(company_of_interest,aspect_of_interest)],all_sentence[(company,avg)])
            except:
                print(f'{company} had no review')
    
    result = [(k,v) for k, v in sorted(mine.items(), key=lambda item: item[1], reverse=True)]
    
    # Vanilla doc2vec
    doc2vec_model = Doc2Vec.load(path+f'abae/{text_type}/{text_type}_vanilla_doc2vec_model')
    l = []
    for (c, s) in doc2vec_model.docvecs.most_similar(company_of_interest, topn=606):
        #l.append((c,s))
        if c in companies:
            l.append((c,s))
    
    print('Mine:')
    print(result)
    print('Vanilla:')
    print(l)


### Make tf-idf
#df = torch.load('../sample_data/master/sentence_gics.pt')
#aspect = 'People and Culture'

#final = df.to_dict('records')

tfidf = torch.load(path+'abae/pros/aspect_size_12/master_tf_idf.pt')
tfidf = tfidf[tfidf['aspect_1']==aspect_of_interest]
tfidf = calculate_tf_idf(tfidf)
tfidf['conml'] = tfidf.index

test = tfidf[tfidf['conml'].isin(companies)]
test['amazing']


test[test['conml']==company_of_interest].transpose().drop(['conml']).sort_values(by=company_of_interest).tail(30)

###