# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)


# Parameters
aspect = 'SeniorLeadership' # CultureAndValues CompensationAndBenefits WorkLifeBalance CareerOpportunities

labeled_pros = torch.load('../sample_data/master/pros_12_aspect_labeled.pt')
labeled_pros = labeled_pros[labeled_pros['aspect']==aspect]
labeled_pros = labeled_pros.reset_index(drop=True)

labeled_cons = torch.load('../sample_data/master/cons_12_aspect_labeled.pt')
labeled_cons = labeled_cons[labeled_cons['aspect']==aspect]
labeled_cons = labeled_cons.reset_index(drop=True)

origin = torch.load('../sample_data/master/review_metadata.pt')[['reviewId', 'company', 'rating'+aspect]]
origin = origin[origin.groupby('company')['reviewId'].transform('count') > 50]
origin = origin[origin['rating'+aspect]>0]
origin = origin.reset_index(drop=True)
print(f"Average of all reviews: {np.mean(origin['rating'+aspect])}")

merged_cons = pd.merge(labeled_cons, origin[['reviewId', 'rating'+aspect]], on='reviewId')
print(f"Average of all cons: {np.mean(merged_cons['rating'+aspect])}")

merged_pros = pd.merge(labeled_pros, origin[['reviewId', 'rating'+aspect]], on='reviewId')
print(f"Average of all pros: {np.mean(merged_pros['rating'+aspect])}")

duplicates = pd.merge(labeled_pros, labeled_cons, on='reviewId')
print(f'{len(duplicates)} overlap out of {len(origin)} ({len(duplicates)/len(origin)*100}%)')



# Per-Company Analysis
rating_all = pd.DataFrame(origin.groupby('company').mean('rating'+aspect)['rating'+aspect])
rating_pros = pd.DataFrame(merged_pros.groupby('company').mean(aspect)['rating'+aspect])
rating_pros.rename({'rating'+aspect: 'rating'+aspect+'_adjusted'}, axis=1, inplace=True)
rating_cons = pd.DataFrame(merged_cons.groupby('company').mean(aspect)['rating'+aspect])
rating_cons.rename({'rating'+aspect: 'rating'+aspect+'_adjusted'}, axis=1, inplace=True)

final = pd.concat([rating_all, rating_pros, rating_cons], axis=1)
final['ratingDifference'] = final['rating'+aspect+'_adjusted'] - final['rating'+aspect]

diffsum = sum([float(x) for x in final['rating'+'Difference'] if ~np.isnan(x)])
print(diffsum)
print(diffsum/len(final))

# Random
print(sum(origin.sample(n=835557).groupby('company').mean('rating'+aspect)['ratingCultureAndValues']))

print(sum(final['rating'+aspect+'_adjusted']))