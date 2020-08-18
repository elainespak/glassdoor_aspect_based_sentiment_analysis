# -*- coding: utf-8 -*-

import os
import argparse
import logging
import numpy as np
from time import time
import utils as U
from sklearn.metrics import classification_report
import codecs
from keras.preprocessing import sequence
import reader as dataset
from model import create_model
import keras.backend as K
from optimizers import get_optimizer


def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)

def evaluation(true, predict, domain):
    true_label = []
    predict_label = []

    if domain == 'restaurant':

        for line in predict:
            predict_label.append(line.strip())

        for line in true:
            true_label.append(line.strip())

        print(classification_report(true_label, predict_label, 
            ['Food', 'Staff', 'Ambience', 'Anecdotes', 'Price', 'Miscellaneous'], digits=3))

    else:
        for line in predict:
            label = line.strip()
            if label == 'smell' or label == 'taste':
              label = 'taste+smell'
            predict_label.append(label)

        for line in true:
            label = line.strip()
            if label == 'smell' or label == 'taste':
              label = 'taste+smell'
            true_label.append(label)

        print(classification_report(true_label, predict_label, 
            ['feel', 'taste+smell', 'look', 'overall', 'None'], digits=3))


#def prediction(test_labels, aspect_probs, cluster_map, domain):
#    label_ids = np.argsort(aspect_probs, axis=1)[:,-1]
#    predict_labels = [cluster_map[label_id] for label_id in label_ids]
#    evaluation(open(test_labels), predict_labels, domain)

def prediction(aspect_probs, domain):
    label_ids = np.argsort(aspect_probs, axis=1)[:,-3:]
    label_probs = np.sort(aspect_probs, axis=1)[:,-3:]
    with open(out_dir+'/predicted_aspects.txt', 'w') as f:
        for i, p in zip(label_ids, label_probs):
            f.write(str(i[-1]))
            f.write(',')
            f.write(str(p[-1]))
            f.write(',')
            f.write(str(i[-2]))
            f.write(',')
            f.write(str(p[-2]))
            f.write(',')
            f.write(str(i[-3]))
            f.write(',')
            f.write(str(p[-3]))
            f.write('\n')

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training

aspect_size = 20
pre_dir = '../output_dir/glassdoor/aspect_size_' + str(aspect_size)
out_dir = '../output_dir/glassdoor/aspect_size_' + str(aspect_size) + '/tests'
algorithm = 'adam'
domain = 'glassdoor'
vocab_size = 9000 # '0' means no limit (default=9000)
maxlen = 0 # Maximum allowed number of words during training. '0' means no limit (default=0)
ortho_reg = 0.1
neg_size = 20
emb_dim = 200
emb_path = r'../preprocessed_data/glassdoor/w2v_embedding'

assert algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}

# map for the pre-trained glassdoor model (pros, cons, advice / trigram / 20 aspects)
cluster_map = {0: 'None', 1: 'Culture', 2: 'Perks', 3: 'Technical',
               4: 'Overall', 5: 'Benefits', 6:'None',  7: 'Restructuring', 8: 'Structure and Policies', 
               9: 'Customers, Products, and Services', 10: 'Moral Values', 11: 'Workspace', 
               12: 'Working Conditions', 13: 'Senior Leadership', 14: 'Work Life Balance',
               15: 'Location', 16: 'Career Opportunities: Junior Perspective',
               17: 'Compensation', 18: 'Career Opportunities: Senior Perspective',
               19: 'People'}
    
###### Get test data #############
#filename = r'gold/sentences/all_tokenized_trigram_sentences_000'
files = os.listdir(r'gold/sentences/')
for filename in files:

    vocab, train_x, test_x, overall_maxlen = dataset.get_data(domain,
                                                              r'gold/sentences/'+filename,
                                                              vocab_size=vocab_size,
                                                              maxlen=maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)


    ############# Build model architecture, same as the model used for training #########
    optimizer = get_optimizer('adam')
    model = create_model(ortho_reg, neg_size, emb_dim, aspect_size, emb_path, overall_maxlen, vocab)
    
    ## Load the save model parameters
    model.load_weights(pre_dir+'/model_param')
    model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])
    
    ################ Evaluation ####################################
    ## Create a dictionary that map word index to word 
    vocab_inv = {}
    for w, ind in vocab.items():
        vocab_inv[ind] = w
    
    test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()], 
            [model.get_layer('att_weights').output, model.get_layer('p_t').output])
    att_weights, aspect_probs = test_fn([test_x, 0])
    ### My code ###
    label_ids = np.argsort(aspect_probs, axis=1)[:,-3:]
    label_probs = np.sort(aspect_probs, axis=1)[:,-3:]
    #############
    
    ## Save attention weights on test sentences into a file 
    att_out = codecs.open(out_dir + '/att_weights_' + filename + '.txt', 'w', 'utf-8')
    print 'Saving attention weights on test sentences...'
    #print('Saving attention weights on test sentences...')
    for c in xrange(len(test_x)):
        att_out.write('----------------------------------------\n')
        att_out.write(str(c) + '\n')
    
        word_inds = [i for i in test_x[c] if i!=0]
        line_len = len(word_inds)
        weights = att_weights[c]
        weights = weights[(overall_maxlen-line_len):]
    
        words = [vocab_inv[i] for i in word_inds]
        ### My code ###
        att_out.write(cluster_map[label_ids[c][-1]] + ': ' + str(label_probs[c][-1]) + '\n' +
                         cluster_map[label_ids[c][-2]] + ': ' + str(label_probs[c][-2]) + '\n' +
                         cluster_map[label_ids[c][-3]] + ': ' + str(label_probs[c][-3]) + '\n')
    
        pass
        #############
        att_out.write(' '.join(words) + '\n')
        for j in range(len(words)):
            att_out.write(words[j] + ' '+str(round(weights[j], 3)) + '\n')



######################################################
# Uncomment the below part for F scores
######################################################

## cluster_map need to be specified manually according to the top words in each inferred aspect (save in aspect.log)

# map for the pre-trained restaurant model (under pre_trained_model/restaurant)
# cluster_map = {0: 'Food', 1: 'Miscellaneous', 2: 'Miscellaneous', 3: 'Food',
#            4: 'Miscellaneous', 5: 'Food', 6:'Price',  7: 'Miscellaneous', 8: 'Staff', 
#            9: 'Food', 10: 'Food', 11: 'Anecdotes', 
#            12: 'Ambience', 13: 'Staff'}


print '--- Results on %s domain ---' % (domain)
#prediction(aspect_probs, domain=domain)


