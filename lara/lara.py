# -*- coding: utf-8 -*-

import torch
import pickle
import string
from nltk import FreqDist
from text_preprocessing import *
from aspect_augmentation import *

maketrans = ''.maketrans
replace_punctuation = maketrans(string.punctuation, ' '*len(string.punctuation))


def main(path, input_file, text_type=['summary', 'pros', 'cons', 'advice']):
    
    raw_text = load_only_text(path + input_file,
                              text_type=['summary','pros','cons','advice'],
                              company=False)
    
    tokenized_sentences = preprocess_word_tokenize(raw_text, replace_punctuation)
    torch.save(tokenized_sentences, path + 'english_glassdoor_reviews_preprocessed_sentences.pt')
    del raw_text
    print('--------------------------- Finished preprocessing and tokenizing sentences into words -')
    
    # Create bigram and trigram models and process sentences with them
    b_model, t_model = make_ngrams_model(tokenized_sentences, 5, 100)
    bigram_sentences, trigram_sentences = make_ngrams(bigram_mod=b_model,
                                                      trigram_mod=t_model,
                                                      tokenized_sents=tokenized_sentences)
    torch.save(bigram_sentences, path + 'english_glassdoor_reviews_english_bigram_sentences.pt')
    torch.save(bigram_sentences, path + 'english_glassdoor_reviews_english_triigram_sentences.pt')
    print('--------------------------- Finished applying n-grams to the tokenized sentences -------')
    
    stemmed_sentences = stemming(bigram_sentences)
    torch.save(stemmed_sentences, path + 'english_glassdoor_reviews_english_stemmed_sentences.pt')
    print('--------------------------- Finished stemming ------------------------------------------')
    
    # Create vocabs list and vocabs dictionary
    vocab, vocab_dict = create_vocab(stemmed_sentences)
    torch.save(vocab, path + 'english_glassdoor_reviews_english_vocab.pt')
    torch.save(vocab, path + 'english_glassdoor_reviews_english_vocab_dict.pt')
    print('--------------------------- Finished saving vocabs -------------------------------------')


if __name__ == "__main__":
    
    main('../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/',
         'english_glassdoor_reviews.pt')
    """
    ### ------------------------------------ Check how frequent the bigrams are!
    test = to_one_list(only_sent)
    test_freq = FreqDist(test)
    ok = sorted([(test_freq[k],k) for k,v in vocab_dict.items() if '_' in k], reverse=True)
    print(ok[:15])
    ### ------------------------------------------------------------------------
    
    
    
    
    ## == 0. Setup =================================================================
    
    reviews_path = '../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/english_glassdoor_reviews.pt'
    output_path = '../sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/'
    text_type = ['summary', 'pros', 'cons', 'advice']
    
    
    
    ## == 1. Create VocabDict ======================================================
    
    replace_punctuation = maketrans(string.punctuation, ' '*len(string.punctuation))
    all_review_processed, all_only_sent = preprocess_word_tokenize(all_reviews, replace_punctuation)
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
    
    """