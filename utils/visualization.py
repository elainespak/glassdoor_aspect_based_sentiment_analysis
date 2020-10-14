# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)

origin = torch.load('../sample_data/master/review_metadata.pt')

# Average num. of reviews per company
avg = origin.groupby('company').count()['featured']
np.mean(avg)

# Avg. char. count in pros vs. cons
avg_pros = origin.pros.apply(lambda x: len(' '.join(x)))
np.mean(avg_pros)

avg_cons = origin.cons.apply(lambda x: len(' '.join(x)))
np.mean(avg_cons)

avg_advice = origin.advice.apply(lambda x: len(' '.join(x)))
np.mean(avg_advice)

# Missing rating data portion
aspects = ['CompensationAndBenefits','WorkLifeBalance','CareerOpportunities','CultureAndValues','SeniorLeadership',
           'BusinessOutlook','Ceo','RecommendToFriend']

for aspect in aspects:
    print(aspect)
    print(len(origin[origin['rating'+aspect]<0.5]))