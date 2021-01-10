# -*- coding: utf-8 -*-

from tools import *

### Parameters
#alldat = pd.read_csv('2018_SnP500_Names_Subsidiaries.csv', delimiter=',')
#alldat.Company
names = ['EBS Healthcare'] ### !!! ### Define your own company list
driver_path = r'C:\Users\elain\chromedriver.exe' ### !!! ### Define your own chromedriver path


### Runs fine except for the "big companies"
company_tokens = {}

for name in names:
    
    # Generate the company-specific part of the url
    company_token = get_company_token(name, driver_path)
    company_tokens[name] = company_token
    #company_token = 'Scripps-Networks-Interactive-Reviews-E38201'
    #company_token = 'First-Republic-Bank-Reviews-E859'
    
    if company_token == 'notoken':
        # Print if no Glassdoor reviews exist
        print('No.' + str(list(names).index(name)) + ' company "' + name + '" reviews do not exist')
    else:
        
        # Access the Glassdoor reviews page
        url = 'https://www.glassdoor.com/Reviews/' + company_token
        page_number = ''
        num = 2
        html = '.htm'
        page = requests.get(url+page_number+html, headers={'user-agent': 'Mozilla/5.0'})
        webpage = page.text

        # First, get the overall stats
        overall_stats = {name: get_overall_stats(webpage)}

        # Save the overall stats data into a text file
        with open(re.sub(' ', '_', name)+'_overall_stats.txt', 'w') as file:
             file.write(json.dumps(overall_stats))

        # Secondly, get the individual reviews
        individual_reviews = {dct['reviewId']: dct for dct in get_individual_reviews(webpage)} 
        end_of_reviews = False

        while end_of_reviews is False:
            page_number = '_P' + str(num) # Go to the next review page
            page = requests.get(url+page_number+html, headers={'user-agent': 'Mozilla/5.0'})
            webpage = page.text

            if 'reviewid=' not in webpage: # If we run out of pages, exit the while loop
                end_of_reviews = True
            else:
                for dct in get_individual_reviews(webpage):
                    individual_reviews[dct['reviewId']] = dct

                if num%100 == 0: # If we reach page 300, save the crawled data into a txt file and empty it
                    with open(re.sub(' ', '_', name)+'_individual_reviews_'+str(int(num/100))+'.txt', 'w') as file:
                        file.write(json.dumps(individual_reviews))
                    individual_reviews.clear()
                    individual_reviews = {dct['reviewId']: dct for dct in get_individual_reviews(webpage)} 

                num += 1

            time.sleep(randint(2,3))

        # Print if successfully scraped all the reviews
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
    time.sleep(4)