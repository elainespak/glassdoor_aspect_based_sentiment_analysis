# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from doc2vec import *
from tf_idf_keyword_extraction import calculate_tf_idf
pd.set_option('display.max_columns', 50)


# Parameters
text_type = 'pros'
path = f'../sample_data/master/{text_type}_12_aspect_labeled.pt'
gics_path = '../sample_data/master/company_gics.pt'
tokens = 'trigramSentence'
tags = 'company'

aspects = ['CultureAndValues', 'CompensationAndBenefits', 'CareerOpportunities',
           'BusinessOutlook', 'SeniorLeadership', 'WorkLifeBalance']    

# Bring vanilla doc2vec embeddings
aspect = None
vanilla_model_path =  f'../sample_data/abae/{text_type}/{text_type}_vanilla_doc2vec_model'
vanilla = Doc2VecEmbeddings(path, vanilla_model_path, aspect, tokens, tags)
vanilla.get_model()

# Bring ABAE-based aspect-specific doc2vec embeddings
aspect = 'CompensationAndBenefits'
model_path = f'../sample_data/abae/{text_type}/{text_type}_{aspect}_doc2vec_model'
embeddings = Doc2VecEmbeddings(path, model_path, aspect, tokens, tags)
embeddings.get_model()


# GICS details
group = 'gind' # 'ggroup'
group_number = 401010
company_of_interest = 'Citigroup_Inc'

# Compare
print(f'------------- ABAE {aspect}: \n')
print(embeddings.get_most_similar_companies('../sample_data/master/company_gics.pt', group, group_number, company_of_interest))
print('\n\n')
print('------------- Vanilla: \n')
print(vanilla.get_most_similar_companies('../sample_data/master/company_gics.pt', group, group_number, company_of_interest))


# Quantitative comparison



# Cosine similarity
tfidf = torch.load('../sample_data/abae/pros/aspect_size_12/master_tf_idf.pt')
tfidf = tfidf[tfidf['aspect_1']==aspect_of_interest]
tfidf = calculate_tf_idf(tfidf)
tfidf['conml'] = tfidf.index

test = tfidf[tfidf['conml'].isin(companies)]
test['amazing']


test[test['conml']==company_of_interest].transpose().drop(['conml']).sort_values(by=company_of_interest).tail(30)
