# -*- coding: utf-8 -*-

import os
import ast
import pickle
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
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='pros')
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000, help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1, help="The weight of orthogonol regularizaiton (default=0.1)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20, help="Number of negative instances (default=20)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14, help="The number of aspects specified by users (default=14)")

parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200, help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")

args = parser.parse_args()
assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}


def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)


######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training

pre_dir = '../sample_data/abae/'+args.domain+'/aspect_size_' + str(args.aspect_size)
out_dir = pre_dir + '/tests_results'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# map for the pre-trained glassdoor model (pros, cons, advice / trigram / 20 aspects)

with open(pre_dir+'/cluster_map.txt', 'r') as f:
    cluster_map = f.readlines()
cluster_map = ''.join([i.replace('\n', '') for i in cluster_map])
cluster_map = ast.literal_eval(cluster_map)
print(type(cluster_map))
print(cluster_map)
    
###### Get test data #############
filepath = '../sample_data/abae/'+args.domain+'/tests/'
files = os.listdir(filepath)
for filename in tqdm(files):
    vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, 'tests/'+filename.split('.t')[0], vocab_size=args.vocab_size, maxlen=args.maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)

    ############# Build model architecture, same as the model used for training #########
    optimizer = get_optimizer(args)
    model = create_model(args, overall_maxlen, vocab)
    
    ## Load the save model parameters
    model.load_weights(pre_dir+'/model_param')
    model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])
    
    ################ Evaluation ####################################
    ## Create a dictionary that map word index to word 
    vocab_inv = {}
    for w, ind in vocab.items():
        vocab_inv[ind] = w
    
    test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()], 
            [model.get_layer('att_weights').output, model.get_layer('y_s').output, model.get_layer('z_s').output, model.get_layer('p_t').output])
    att_weights, unweighted_sentence_embeddings, sentence_embeddings, aspect_probs = test_fn([test_x, 0])
    label_ids = np.argsort(aspect_probs, axis=1)[:,-3:]
    label_probs = np.sort(aspect_probs, axis=1)[:,-3:]

    # Save attention weights on test sentences into a file
    final_info = []
    for c in xrange(len(test_x)):
        d = {}
        word_inds = [i for i in test_x[c] if i!=0]
        d['words'] = [vocab_inv[i] for i in word_inds]
        d['unweighted_sentence_embedding'] = unweighted_sentence_embeddings[c]
        d['attention_weights'] = att_weights[c][(overall_maxlen-len(word_inds)):]
        d['sentence_embedding'] = sentence_embeddings[c]
        d['aspect_1'] = cluster_map[label_ids[c][-1]]
        d['aspect_1_prob'] = label_probs[c][-1]
        d['aspect_2'] = cluster_map[label_ids[c][-2]]
        d['aspect_2_prob'] = label_probs[c][-2]
        d['aspect_3'] = cluster_map[label_ids[c][-3]]
        d['aspect_3_prob'] = label_probs[c][-3]
        final_info.append(d)

    with open(out_dir+'/test_result_'+filename.split('.t')[0]+'.pickle', 'w') as f:
        pickle.dump(final_info, f)
