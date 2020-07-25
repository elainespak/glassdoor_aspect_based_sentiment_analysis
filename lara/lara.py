# -*- coding: utf-8 -*-

import os
import pickle
import string
from tqdm import tqdm
from nltk import FreqDist
from text_preprocessing import *
from aspect_augmentation import *
maketrans = ''.maketrans


if __name__ == "__main__":
    
    
    ## == 0. Setup =================================================================
    
    path = '../sample_data/2008 to 2018 SnP 500 Firm Data All/'
    output_path = '../sample_output/lara/'
    text_type = ['summary', 'pros', 'cons', 'advice']
    
    
    ## == 1. Create VocabDict ======================================================
    
    all_reviews = []
    company_list = []
    for file in tqdm(os.listdir(path)):
        temp_reviews = load_file(path+file, text=text_type)
        print(f'{file} successfully loaded')
        company_name = file.split('_individual_reviews')[0]
        company_name = company_name.replace('_', ' ')
        company_list.append(company_name)
        for r in temp_reviews:
            all_reviews.append(r)
    ### Save ###
    with open('../sample_output/lara/raw_english_sentences.pkl', 'wb') as f:
        pickle.dump(all_reviews, f)
    print('--------------------------- Finished loading all files -----------')
    
    replace_punctuation = maketrans(string.punctuation, ' '*len(string.punctuation))
    all_review_processed, all_only_sent = parse_all_reviews_to_sentence(all_reviews, replace_punctuation)
    ### Save ###
    with open('../sample_output/lara/processed_english_sentences.pkl', 'wb') as f:
        pickle.dump(all_only_sent, f)
    with open('../sample_output/lara/processed_english_sentences_per_review.pkl', 'wb') as f:
        pickle.dump(all_review_processed, f)
    print('--------------------------- Finished parsing reviews to sentences -')
    
    # Create bigram and trigram models and process sentences with them
    b_model, t_model = make_ngrams_model(all_only_sent, 5, 20)
    all_only_sent, _ = make_ngrams(bigram_mod=b_model, trigram_mod=t_model, tokenized_sents=all_only_sent)
    ### Save ###
    with open('../sample_output/lara/processed_english_sentences.pkl', 'wb') as f:
        pickle.dump(all_only_sent, f)
    print('--------------------------- Finished creating n grams ------------')
    
    # Stemming
    only_sent = stemming(all_only_sent)
    print('--------------------------- Finished stemming --------------------')
    
    # Create vocabs list and vocabs dictionary
    vocab, vocab_dict = create_vocab(only_sent)
    ### Save ###
    with open('../sample_output/lara/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('../sample_output/lara/vocab_dict.pkl', 'wb') as f:
        pickle.dump(vocab_dict, f)
    print('--------------------------- Finished creating vocab & vocab dict -')
    
    ### ------------------------------------ Check how frequent the bigrams are!
    test = to_one_list(only_sent)
    test_freq = FreqDist(test)
    ok = sorted([(test_freq[k],k) for k,v in vocab_dict.items() if '_' in k], reverse=True)
    print(ok[:15])
    ### ------------------------------------------------------------------------
    
    
    ## == 2. Select keywords per aspect ============================================
    
    analyzer = Bootstrapping()
    
    # Load aspect seedwords
    load_Aspect_Terms(analyzer, path + 'aspect_seed_words_bigrams.txt', vocab_dict)
    for aspect in analyzer.Aspect_Terms:
        print('-------- Aspect Seedwords:')
        print(aspect)
        print([vocab[w] for w in aspect])
    
    # Define corpus (test with two companies)
    data = Corpus(company_list, vocab, vocab_dict)
    
    # Labeling each sentence
    analyzer.sentence_label(data)
    
    # Calculate chi square
    analyzer.calc_chi_sq(data) # it works! CHECK LATER to see if +0.00001 is justified
    
    # Update the aspect keywords list
    load_Aspect_Terms(analyzer, path + 'aspect_seed_words_bigrams.txt', vocab_dict)
    Add_Aspect_Keywords(analyzer, p=5, NumIter=5, c=data)
    
    # Save the aspect keywords
    aspectfile = output_path + "aspect_final_words_bigrams_8.txt"
    f = open(aspectfile, 'w',encoding='UTF-8')
    
    for aspect in analyzer.Aspect_Terms:
        print('-------- Final Aspect terms:')
        for w in aspect:
            print(vocab[w])
            f.write(vocab[w])
            f.write(',')
        f.write('\n')
    f.close()
    
    # Create W matrix for each review
    create_all_W(analyzer,data)
    
    # W matrix for all reviews
    produce_data_for_rating(analyzer, data, output_path, percompany=False)
    
    # W matrix for reviews per company
    produce_data_for_rating(analyzer, data, output_path, percompany=True)