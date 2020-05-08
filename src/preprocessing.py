# -*- coding: utf-8 -*-

import json
import pandas as pd
import shutil
import os
from os import path

# == Add files and move them to proper folder =============================================

def merge_individual_to_all(company_names, folder):
    for name in company_names:
        alldat = {}
        for i in list(range(1,30))+['end']:
            if i != 'end':
                try:
                    alldat.update(json.load(open(folder+name.replace(' ','_')+'_individual_reviews_'+str(i)+'.0.txt')))
                except:
                    pass
            else:
                try:
                    alldat.update(json.load(open(folder+name.replace(' ','_')+'_individual_reviews_end.txt')))
                except:
                    pass
        if alldat != {}:
            with open(folder+name.replace(' ','_')+'_individual_reviews_all.txt', 'w') as file:
                file.write(json.dumps(alldat))

def move_all_to_final_folder(source_folder, destination_folder):
    files = [i for i in os.listdir(source_folder) if i.endswith("_all.txt") and path.isfile(path.join(source_folder, i))]
    for f in files:
        shutil.copy(path.join(source_folder, f), destination_folder)


# Set year parameter
year = '2008'

# Bring in the company names.
csv_name = 'data/2008_to_2018_SnP500_Names.csv'
allcompanies = pd.read_csv(csv_name, delimiter=',')
names = allcompanies['conml']



# == Check S&P constituent status by quarter ==============================================

def get_from_quarter(x):
    '''
    change 'from' column date to year-quarter format
    '''
    if 1 <= int(x[5:7]) <= 3:
        return x[:4] + '-Quarter2'
    elif 4 <= int(x[5:7]) <= 6:
        return x[:4] + '-Quarter3'
    elif 6 <= int(x[5:7]) <= 9:
        return x[:4] + '-Quarter4'
    else:
        return str(int(x[:4])+1) + '-Quarter1'
    
    
def get_thru_quarter(x):
    '''
    change 'thru' column date to year-quarter format
    '''
    if pd.isnull(x):
        return '2019-Quarter3' # The most current year-quarter
    elif 1 <= int(x[5:7]) <= 3:
        return x[:4] + '-Quarter1'
    elif 3 <= int(x[5:7]) <= 6:
        return x[:4] + '-Quarter2'
    elif 6 <= int(x[5:7]) <= 9:
        return x[:4] + '-Quarter3'
    else:
        return x[:4] + '-Quarter4'


# Merge all review text files into one file
merge_individual_to_all(names, '2008 to 2018 SnP 500 Firm Data Before All/')
move_all_to_final_folder('2008 to 2018 SnP 500 Firm Data Before All', '2008 to 2018 SnP 500 Firm Data All')


# Call S&P constituents data
allsnp = pd.read_csv('data/S&P500_constituents.csv', delimiter=',')

# Call the mapping data
mapdat = pd.read_csv('data/ticker_mapping.csv', delimiter=',')
mapdat = mapdat.drop_duplicates(['gvkey','conml']).reset_index(drop=True)[['gvkey','conml']]
mapdat.head(3)

# Merge
allsnp = allsnp.merge(mapdat, how='left', left_on='gvkey', right_on='gvkey')
allsnp.head()

# Create S&P 500 year columns
years = list(range(2008,2020))
for year in years:
    allsnp[year] = 0

# Mark the years that the company belonged to in S&P 500 list
for idx in range(len(allsnp)):
    
    if pd.isnull(allsnp['thru'].iloc[idx]): # If the company is still S&P now
        
        if int(allsnp.loc[idx,'from'][:4]) < 2008: # and if company joined S&P before 2008
            allsnp.loc[idx,years] = 1 # mark all years as 'yes' (=1)
        else:
            entering_year = int(allsnp.loc[idx,'from'][:4]) # and if company joined S&P after 2008
            applicable_years = list(range(entering_year,2020))
            allsnp.loc[idx, applicable_years] = 1 # only mark the applicable years as 'yes' (=1)

    elif int(allsnp.loc[idx,'thru'][:4]) >= 2008:
        
        if int(allsnp.loc[idx,'from'][:4]) < 2008:
            entering_year = 2008
        else:
            entering_year = int(allsnp.loc[idx,'from'][:4])
            
        ending_year = int(allsnp.loc[idx,'thru'][:4])
        applicable_years = list(range(entering_year,ending_year+1))
        allsnp.loc[idx, applicable_years] = 1

    else:
        pass
