# -*- coding: utf-8 -*-

import os
import json
import fasttext
import pandas as pd
lid_model = fasttext.load_model('lid.176.bin') 


def is_english(sentence):
    """ input sentence should be a string """
    if '__label__en' == lid_model.predict(sentence)[0][0]:
        return True
    else:
        return False


def create_masterfile(path, output_path):
    all_reviews = []
    not_english_reviews = {}
    
    for file in os.listdir(path):
        f = json.load(open(file))
        reviews_dict = list(f.values())
        
        company = file.split('_individual_reviews_all.txt')[0]
        not_english_reviews[company] = 0
        print(f'{company} starts! {len(reviews_dict)} reviews in raw data --------------------')
        
        for review_dict in reviews_dict:
            
            # Check if a review is written in English
            all_text = ['summary', 'pros', 'cons', 'advice']
            length_text = [len(review_dict[t]) for t in all_text]
            idx = length_text.index(max(length_text))
            if is_english(review_dict[all_text[idx]]) is True:
                
                # Add company name
                review_dict['company'] = company
                # Fix location
                if review_dict['Location'] == []:
                    review_dict['locationType'] = ''
                    review_dict['locationName'] = ''
                else:
                    review_dict['locationType'] = review_dict['Location']['locationType']
                    review_dict['locationName'] = review_dict['Location']['locationName']
                # Fix employer responses
                review_dict['employerResponses'] = 'linebreakanotherresponse'.join(review_dict['employerResponses'])
                # Delete the old unnecessary key:value pairs
                del review_dict['employer'], review_dict['Location']
                
                # Save every 50000 reviews
                all_reviews.append(review_dict)
                i = 0
                if len(all_reviews) % 50000 == 0:
                    reviews_df = pd.DataFrame(all_reviews)
                    reviews_df.to_csv(output_path + 'english_glassdoor_reviews_' + str(i) + '.pt')
                    all_reviews = []
                    i += 1
                    print(f'{str(i * 50000)} English reviews saved so far!')
                            
            else:
                not_english_reviews[company] += 1
        
        # Save remaining reviews
        reviews_df = pd.DataFrame(all_reviews)
        reviews_df.to_csv(output_path + 'english_glassdoor_reviews_' + str(i) + '.pt')
        all_reviews = []
        print(f'{str(i * 50000 + len(all_reviews))} English reviews saved so far!')
        print(f' *** For {company}, {not_english_reviews[company]} number of reviews were not in English')
        print('\n')
    pd.DataFrame.from_dict(not_english_reviews, orient='index').to_csv(output_path + 'not_english_glassdoor_reviews.pt')
    print('DONE, all master files created')
    
