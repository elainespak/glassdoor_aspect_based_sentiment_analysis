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
    not_english_reviews = {}
    i = 1
    
    for file in os.listdir(path):
        f = json.load(open(path+file))
        reviews_dict = list(f.values())
        
        company = file.split('_individual_reviews_all.txt')[0]
        not_english_reviews[company] = 0
        print(f'{company} starts! {len(reviews_dict)} reviews in raw data --------------------')
        
        for review_dict in reviews_dict:
            
            # Check if a review is written in English
            all_type = [t for t in ['summary', 'pros', 'cons', 'advice'] if review_dict[t] is not None]
            length_text = [len(review_dict[t]) for t in all_type]
            idx = length_text.index(max(length_text))
            text = review_dict[all_type[idx]].replace('\n', '')
            if is_english(text) is True:
                
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
                
                if len(all_reviews) % 50000 == 0:
                    reviews_df = pd.DataFrame(all_reviews)
                    torch.save(reviews_df, output_path + 'english_glassdoor_reviews_' + str(i) + '.pt')
                    all_reviews = []
                    i += 1
                    print(f'{str(i * 50000)} English reviews saved so far!')
                            
            else:
                not_english_reviews[company] += 1
        
        # Save remaining reviews
        reviews_df = pd.DataFrame(all_reviews)
        torch.save(reviews_df, output_path + 'english_glassdoor_reviews_' + str(i) + '.pt')
        all_reviews = []
        print(f'{str(len(all_reviews))} English reviews saved so far!')
        print(f' *** For {company}, {not_english_reviews[company]} number of reviews were not in English')
        print('\n')
    not_english_df = pd.DataFrame.from_dict(not_english_reviews, orient='index')
    not_english_df.columns = ['non_english_reviews_number']
    torch.save(not_english_df, output_path + 'not_english_glassdoor_reviews.pt')
    print('DONE, all master files created')
    

if __name__ == "__main__":
    create_masterfile('./Desktop/glassdoor_aspect_based_sentiment_analysis/scraped_data/2008 to 2018 SnP 500 Firm Data All/',
                      './Desktop/glassdoor_aspect_based_sentiment_analysis/sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/')