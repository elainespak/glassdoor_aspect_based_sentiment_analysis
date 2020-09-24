# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from doc2vec import *
from tqdm import tqdm
from tf_idf_keyword_extraction import calculate_tf_idf
pd.set_option('display.max_columns', 50)


# Parameters
text_type = 'all'
path = f'../sample_data/master/{text_type}_12_aspect_labeled.pt'
gics_path = '../sample_data/master/company_gics.pt'
tokens = 'trigramSentence'
tags = 'company'

# Bring vanilla doc2vec embeddings
aspect = None
vanilla_model_path =  f'../sample_data/abae/{text_type}/{text_type}_vanilla_doc2vec_model'
vanilla_model_path = '../sample_data/master/all_vanilla_doc2vec_model'
vanilla = Doc2VecEmbeddings(path, vanilla_model_path, aspect, tokens, tags)
vanilla.get_model()

# Bring ABAE-based aspect-specific doc2vec embeddings
aspect = 'SeniorLeadership'
model_path = f'../sample_data/abae/{text_type}/{text_type}_{aspect}_doc2vec_model'
model_path = f'../sample_data/master/all_{aspect}_doc2vec_model'
embeddings = Doc2VecEmbeddings(path, model_path, aspect, tokens, tags)
embeddings.get_model()


# GICS details
group = 'ggroup' # 'gind'
group_number = 1010
company_of_interest = 'Comerica_Inc'

# Compare
print(f'------------- ABAE {aspect}: \n')
embeddings.get_gics_companies('../sample_data/master/company_gics.pt', group, group_number)
print(embeddings.get_most_similar_companies(company_of_interest))
print('\n\n')
print('------------- Vanilla: \n')
vanilla.get_gics_companies('../sample_data/master/company_gics.pt', group, group_number)
print(vanilla.get_most_similar_companies(company_of_interest))


# Quantitative comparison
origin = torch.load('../sample_data/master/review_metadata.pt')

#group = 'ggroup'
#company_gics = [2010, 3510, 3520, 2550, 4510, 5020, 2510, 2530, 4530, 1510, 5510, 4020, 4030, 2030, 6010, 3020, 4520, 1010, 5010, 3030, 4010, 2520, 2020, 3010]

group = 'gind' # 'ggroup'
company_gics = [201050, 201020, 351010, 352010, 255040, 451020, 502020, 201040,
       251010, 451030, 253020, 453010, 151010, 551050, 402030, 403010,
       352030, 151040, 203020, 601010, 352020, 551010, 502030, 255020,
       502010, 302030, 151030, 551030, 402020, 551040, 351020, 452030,
       101020, 101010, 452020, 302020, 201010, 452010, 501010, 551020,
       303020, 401010, 402010, 255030, 302010, 252020, 203010, 253010,
       252030, 201060, 601020, 351030, 303010, 202010, 301010, 203040,
       252010, 202020, 201070, 401020, 201030, 251020, 255010, 151020,
       501020]

aspects = ['CultureAndValues', 'CompensationAndBenefits', 'CareerOpportunities',
               'BusinessOutlook', 'SeniorLeadership', 'WorkLifeBalance']

all_residuals = {}
all_vanilla_residuals = {}

nums = []
vanilla_nums = []
for gic in tqdm(company_gics):
    
    aspect = None
    #vanilla_model_path =  f'../sample_data/abae/{text_type}/{text_type}_vanilla_doc2vec_model'
    vanilla_model_path = '../sample_data/master/all_vanilla_doc2vec_model'
    vanilla = Doc2VecEmbeddings(path, vanilla_model_path, aspect, tokens, tags)
    vanilla.get_model()
    vanilla.get_gics_companies('../sample_data/master/company_gics.pt', group, group_number=gic)
    
    residuals = {}
    vanilla_residuals = {}
    for aspect in aspects:
        #model_path = f'../sample_data/abae/{text_type}/{text_type}_{aspect}_doc2vec_model'
        model_path = f'../sample_data/master/all_{aspect}_doc2vec_model'
        embeddings = Doc2VecEmbeddings(path, model_path, aspect, tokens, tags)
        embeddings.get_model()
        embeddings.get_gics_companies('../sample_data/master/company_gics.pt', group, group_number=gic)
        
        ratings = origin.groupby('company').mean(['rating'+aspect])['rating'+aspect]
        
        companies = embeddings.companies
        for company_of_interest in companies:
            try:
                residuals[(company_of_interest, aspect)] = []
                embeddings.get_most_similar_companies(company_of_interest)
                for company, cosine in embeddings.most_similar_companies[:5]:
                    diff = abs(ratings[company] - ratings[company_of_interest])
                    residuals[(company_of_interest, aspect)].append(diff)
                    nums.append(diff)
                
                vanilla_residuals[(company_of_interest, aspect)] = []
                vanilla.get_most_similar_companies(company_of_interest)
                for company, cosine in vanilla.most_similar_companies[:5]:
                    diff = abs(ratings[company] - ratings[company_of_interest])
                    vanilla_residuals[(company_of_interest, aspect)].append(diff)
                    vanilla_nums.append(diff)
            except:
                print(f'{company_of_interest} error')
    
    all_residuals[gic] = residuals
    all_vanilla_residuals[gic] = vanilla_residuals

diff = [vn - n for n, vn in zip(nums, vanilla_nums)]
np.mean(diff)


list1 = []
for gic, d in all_residuals.items():
    #print(f'{gic}: {sum([sum(v) for v in d.values()])}')
    list1.append(sum([sum(v) for v in d.values()]))
list2 = []
for gic, d in all_vanilla_residuals.items():
    #print(f'{gic}: {sum([sum(v) for v in d.values()])}')
    list2.append(sum([sum(v) for v in d.values()]))

d = {'gics': company_gics,
     'residual': list1, 'vanilla_residual': list2}
df = pd.DataFrame(d)
df['diff'] = df['vanilla_residual'] - df['residual']
np.mean(df['diff'])

