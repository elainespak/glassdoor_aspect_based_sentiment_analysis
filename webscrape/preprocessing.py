# -*- coding: utf-8 -*-

import os
import json
import torch
import fasttext
import pandas as pd
lid_model = fasttext.load_model('lid.176.bin') 
pd.set_option('display.max_columns', 50)


def is_english(sentence):
    """ input sentence should be a string """
    if '__label__en' == lid_model.predict(sentence)[0][0]:
        return True
    else:
        return False


def create_masterfile(path, output_path):
    all_reviews = []
    not_english_reviews = []
    count_number = {}
    
    for file in os.listdir(path):
        f = json.load(open(path+file))
        reviews_dict = list(f.values())
        
        company = file.split('_individual_reviews_all.txt')[0]
        count_number[company] = [0, 0]
        print(f'{company} starts! {len(reviews_dict)} reviews in raw data --------------------')
        
        for review_dict in reviews_dict:
            
            # Fix Nonetype error
            all_type = ['summary', 'pros', 'cons', 'advice']
            for t in all_type:
                if review_dict[t] is None:
                    review_dict[t] = ''
            #all_type = [t for t in ['summary', 'pros', 'cons', 'advice'] if review_dict[t] is not None]
            length_text = [len(review_dict[t]) for t in all_type]
            idx = length_text.index(max(length_text))
            text = review_dict[all_type[idx]].replace('\n', '')
            
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
            if review_dict['employerResponses'] == []:
                review_dict['employerResponse'] = ''
                review_dict['employerJobTitle'] = ''
                review_dict['employerResponseDateTime'] = ''
            else:
                # Fix ast error
                review_dict['employerResponses'] = review_dict['employerResponses'].replace("responseDateTime({\\\"format\\\":\\\"ISO\\\"})", "responseDateTime")
                review_dict['employerResponses'] = json.loads(review_dict['employerResponses'])
                review_dict['employerResponse'] = review_dict['employerResponses']['response']
                review_dict['employerJobTitle'] = review_dict['employerResponses']['userJobTitle']
                review_dict['employerResponseDateTime'] = review_dict['employerResponses']['responseDateTime']
                
            # Delete the old unnecessary key:value pairs
            del review_dict['employer']
            del review_dict['Location']
            del review_dict['employerResponses']
            
            # Check if a review is written in English
            if is_english(text) is True:
                all_reviews.append(review_dict)
                count_number[company][0] += 1
            else:
                not_english_reviews.append(review_dict)
                count_number[company][1] += 1
        
        print(f' *** For {company}, {count_number[company][0]} number of reviews were in English')
        print(f' *** For {company}, {count_number[company][1]} number of reviews were NOT in English')
        print('\n')
    
    # Save english reviews
    reviews_df = pd.DataFrame(all_reviews)
    torch.save(reviews_df, output_path + 'english_glassdoor_reviews.pt')
    print(f'TOTAL ENGLISH REVIEWS: {len(reviews_df)}')
    
    # Save non-english reviews
    not_english_reviews_df = pd.DataFrame(not_english_reviews)
    torch.save(not_english_reviews_df, output_path + 'not_english_glassdoor_reviews.pt')
    print(f'TOTAL NON-ENGLISH REVIEWS: {len(not_english_reviews_df)}')
    
    # Save their counts
    count_number_df = pd.DataFrame.from_dict(count_number, orient='index')
    count_number_df.columns = ['english_count', 'not_english_count']
    torch.save(count_number_df, output_path + 'count_not_english_glassdoor_reviews.pt')
    
    print('DONE, all master files created!!')
    

if __name__ == "__main__":
    create_masterfile('./Desktop/glassdoor_aspect_based_sentiment_analysis/scraped_data/2008 to 2018 SnP 500 Firm Data All/',
                      './Desktop/glassdoor_aspect_based_sentiment_analysis/sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/')