# -*- coding: utf-8 -*-

### Import necessary packages
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import re
import pandas as pd
import requests
import json
from random import randint


def get_company_token(company, driver_path):
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
    driver = webdriver.Chrome(driver_path) # paranthesis is empty since the driver is in the same path
    driver.get('http://google.com/') # it is faster to google than to search for the company on Glassdoor (crappy site)
    
    # Iterate through the list of company names
    search_text = 'glassdoor ' + company + ' Reviews'
    search = driver.find_element_by_name('q')
    search.send_keys(search_text)
    search.send_keys(Keys.RETURN) # hits return after you enter the search text
    time.sleep(2)
    
    #links = driver.find_elements_by_partial_link_text('glassdoor.com/Reviews/')
    links = driver.find_element_by_partial_link_text('Reviews | Glassdoor')
    try:
        full_link = links.get_attribute('href')
        company_token = re.search('(?<=Reviews/)([\w-]*)(?=.htm)',full_link)[1] # the token is between 'Reviews/' and '.htm'
    except:
        print('No.' + str(list(names).index(company)) + ' company "' + company + '" reviews do not exist, sth is off')
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
    review_lst = re.findall('(?<={"isLegal":true,)(.*?)(?={"reviewDetailUrl")', webpage_text) # .*? means "any" ; ?: will prevent interference
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
            temp_dct['employerResponses'] = '{'+re.findall('"response":".*?,"countHelpful":', review_second_half)[0][:-16]+'}'

        review_dct_lst.append(temp_dct)
        
    return review_dct_lst

