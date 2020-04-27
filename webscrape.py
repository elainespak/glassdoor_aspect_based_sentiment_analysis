#!/usr/bin/env python
# coding: utf-8

# Import necessary packages
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import re
import pandas as pd
import requests
import json
from random import randint


# Get the names of the S&P 500 companies
alldat = pd.read_csv('2018_SnP500_Names_Subsidiaries.csv', delimiter=',')
names = alldat.Company

def get_company_token(company):
    '''
    Every company's Glassdoor review page url includes a randomly-assigned token
    (e.g., Booz Allen Hamilton: E2735)
    This is an automated selenium program that collects the token for each of the S&P 500 companies
    '''
    
    # Set up the webdriver (make sure that the driver is saved in the same path as the Jupyter notebook file)
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument("--test-type")
    options.binary_location = "/usr/bin/chromium"
    driver = webdriver.Chrome() # paranthesis is empty since the driver is in the same path
    driver.get('https://www.google.com') # it is faster to google than to search for the company on Glassdoor (crappy site)
    
    # Iterate through the list of company names
    search_text = 'glassdoor ' + company + ' Reviews'
    search = driver.find_element_by_name('q')
    search.send_keys(search_text)
    search.send_keys(Keys.RETURN) # hits return after you enter the search text
    time.sleep(1)
    
    links = driver.find_elements_by_partial_link_text('https://www.glassdoor.com/Reviews/')
    try:
        full_link = links[0].get_attribute('href')
        company_token = re.search('(?<=Reviews/)([\w-]*)(?=.htm)',full_link)[1] # the token is between 'Reviews/' and '.htm'
    except:
        print('No.' + str(list(names).index(name)) + ' company "' + name + '" reviews do not exist, sth is off')
        company_token = 'notoken'
    
    driver.quit()
    
    return company_token


def get_overall_stats(webpage_text):
    '''
    Overall summary of all the reviews posted for the company as of (today)
    '''
    overall_str = re.search('{"overallRating":[^;]*"EmployerRatings"}', webpage_text).group(0)
    overall_dict = json.loads(overall_str)
    
    return overall_dict


def get_individual_reviews(webpage_text):
    '''
    Each review posted for the company as of (today)
    '''
    review_lst = re.findall('(?<={"isLegal":true,)(.*?)(?={"reviewDetailUrl")', apollo) # .*? means "any" ; ?: will prevent interference
    review_lst = [review[:-12]+'}' for review in review_lst] # HARDCODING. need a better solution..
    review_dct_lst = []

    for review in review_lst:
        review_first_half = '{'+review.split(',"links":')[0]+'}'
        review_second_half = review.split(',"links":')[1]

        try:
            temp_dct = json.loads(review_first_half)
        except ValueError:  # Includes simplejson.decoder.JSONDecodeError. Avoids JSONDecode Error: Invalid \escape
            if '\\<' in review:
                temp_dct = json.loads(review_first_half.replace('\\<', '<')) # HARDCODING. need a better solution..
            elif '\\>' in review:
                temp_dct = json.loads(review_first_half.replace('\\>', '>')) # HARDCODING. need a better solution..

        # Look for the reviewer's job title
        reviewer_job = re.findall('(?<=,"text":")(.*?)(?=","__typename":"JobTitle")', review_second_half)
        try:
            temp_dct['jobTitle'] = reviewer_job[0]
        except:
            temp_dct['jobTitle'] = reviewer_job

        # Look for the reviewer's location
        reviewer_loc = re.findall('(?<=,"type":")(.*?)(?=","__typename":"Location)', review_second_half)
        if reviewer_loc == []:
            temp_dct['Location'] = []
        else:
            loc_information = reviewer_loc[0].split('"')
            temp_dct['Location'] = {'locationType': loc_information[0], 'locationName': loc_information[4]}

        # Look for the employer's response to the review, if it exists
        if temp_dct['employerResponses'] != []:
            temp_dct['employerResponses'] = re.findall('{"response":.*?,"countHelpful":', review_second_half)[0][:-16]+'}'

        review_dct_lst.append(temp_dct)
        
    return review_dct_lst


## Runs fine except for the "big companies"

for name in names:
    
    # Generate the company-specific part of the url
    company_token = get_company_token(name)
    #company_token = 'Scripps-Networks-Interactive-Reviews-E38201'
    #company_token = 'First-Republic-Bank-Reviews-E859'
    
    if company_token == 'notoken':
        # Print if no Glassdoor reviews exist
        print('No.' + str(list(names).index(name)) + ' company "' + name + '" reviews do not exist')
    else:
        
        # Avoid max retries error
        #session = requests.Session()
        #retry = Retry(connect=3, backoff_factor=0.5)
        #adapter = HTTPAdapter(max_retries=retry)
        #session.mount('http://', adapter)
        #session.mount('https://', adapter)
        
        # Access the Glassdoor reviews page
        url = 'https://www.glassdoor.com/Reviews/' + company_token
        page_number = ''
        num = 2
        html = '.htm'
        #page = session.get(url+page_number+html, headers={'user-agent': 'Mozilla/5.0'})
        page = requests.get(url+page_number+html, headers={'user-agent': 'Mozilla/5.0'})
        webpage = page.text
        apollo = webpage.split('<script>window.__APOLLO_STATE__')[1]

        # First, get the overall stats
        overall_stats = {name: get_overall_stats(apollo)}

        # Save the overall stats data into a text file
        with open(re.sub(' ', '_', name)+'_overall_stats.txt', 'w') as file:
             file.write(json.dumps(overall_stats))

        # Secondly, get the individual reviews
        individual_reviews = {dct['reviewId']: dct for dct in get_individual_reviews(apollo)} 
        end_of_reviews = False

        while end_of_reviews is False:
            page_number = '_P' + str(num) # Go to the next review page
            page = requests.get(url+page_number+html, headers={'user-agent': 'Mozilla/5.0'})
            webpage = page.text

            if 'reviewId' not in webpage: # If we run out of pages, exit the while loop
                end_of_reviews = True
            else:
                apollo = webpage.split('<script>window.__APOLLO_STATE__')[1]
                for dct in get_individual_reviews(apollo):
                    individual_reviews[dct['reviewId']] = dct

                if num%100 == 0: # If we reach page 300, save the crawled data into a txt file and empty it
                    with open(re.sub(' ', '_', name)+'_individual_reviews_'+str(num/100)+'.txt', 'w') as file:
                        file.write(json.dumps(individual_reviews))
                    individual_reviews.clear()
                    individual_reviews = {dct['reviewId']: dct for dct in get_individual_reviews(apollo)} 

                num += 1

            time.sleep(randint(2,3))

        # Print if successfully scraped all the reviews
        #print('No.' + str(list(names).index(name)) + ' company "' + name + '" reviews all scraped')
        print('Company ' + name + ' reviews all scraped')
        
        # Save the rest of the individual reviews data into a text file
        with open(re.sub(' ', '_', name)+'_individual_reviews_end.txt', 'w') as file:
             file.write(json.dumps(individual_reviews))

        # Empty the dictionarys
        #if bool(individual_reviews):
        #    individual_reviews.clear()
        if bool(overall_stats):
            overall_stats.clear()

    print(str(len(list(names)) - list(names).index(name) - 1) + ' more companies to go')
    print('---------------------------')
    time.sleep(5)




