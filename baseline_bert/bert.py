# -*- coding: utf-8 -*-

import re
import torch
import logging
from tqdm import tqdm
from nltk import sent_tokenize
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel


def load_only_text(master, text_type, company=False):

    if company==False:
        pass
    else:
        # if company is specified, filter for only that company's review data
        master = master[master['company']==company]
        print(f'{company} text loaded!')
    
    # Combine all texts regardless of text_type
    all_sentences = []
    for t in text_type:
        all_sentences += list(master[t])
    
    return all_sentences


def preprocess_sentence_tokenize(raw_sentences):
    """
    ###  INPUT
    # raw_sentences: list of raw sentences
    ###  OUTPUT
    # tokenized_sentences: list of processed sentences
    """
    tokenized_sentences = []
    
    for raw in raw_sentences:
        
        # Change for proper sentence tokenization
        raw = re.sub('\r\n|\n-|\n|\r','. ', raw)
        raw = re.sub(',\.+ ', ', ', raw)
        raw = re.sub('\.+ ', '. ', raw)
        raw = re.sub('&amp;', '&', raw)
        
        # Sentence tokenization
        sentences = sent_tokenize(raw)
        tokenized_sentences += sentences
    return tokenized_sentences


class BERTEmbedding:

    def __init__(self, device, model_type):
         self.device = device
         self.model_type = model_type
         self.model, self.tokenizer = self.retrieve_bert()
         
         # Put the model in "evaluation" mode, meaning feed-forward operation.
         self.model.eval()

    def retrieve_bert(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_type,
                                                  output_hidden_states = True
                                                  )
        model = BertModel.from_pretrained(self.model_type,
                                          # Whether the model returns all hidden-states.
                                          output_hidden_states = True
                                          )
        return model.to(self.device), tokenizer
        #return model, tokenizer

    def get_embeddings(self, sent):
        marked_sent = '[CLS] ' + sent + ' [SEP]'
        
        # Split the sentence into tokens.
        tokenized_sent = self.tokenizer.tokenize(marked_sent)
        if len(tokenized_sent) > 512:
            tokenized_sent = tokenized_sent[:512]

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
        segments_ids = [1] * len(tokenized_sent)
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        # Concatenate the tensors for all layers. `stack` creates a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)
        
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        #print(token_embeddings.size())
        
        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []
        # `token_embeddings` is a [22 x 12 x 768] tensor.
        # For each token in the sentence...
        for token in token_embeddings:
             cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
             token_vecs_cat.append(cat_vec)
        token_vecs = hidden_states[-2][0]
        #print(token_vecs.size())
        
        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        #print(sentence_embedding.size())
        
        return sentence_embedding


if __name__ == "__main__":
     
    # Check for cpu / gpu
    is_cuda = torch.cuda.is_available()
    if is_cuda:
         device = torch.device("cuda")
    else:
         device = torch.device("cpu")
    print(device)
    
    # Set up
    bert_embedding = BERTEmbedding(device, 'bert-base-uncased')
    
    # Call data
    master = torch.load('C:/Users/elain/Desktop/glassdoor_aspect_based_sentiment_analysis/sample_data/2008 to 2018 SnP 500 Firm Data_Master English Files/english_glassdoor_reviews_text_preprocessed.pt')
    all_raw_sentences = load_only_text(master, ['pros', 'cons', 'advice'])
    del master
    print('Done loading data!\n')
    
    all_tokenized_sentences = preprocess_sentence_tokenize(all_raw_sentences)
    print('Done tokenizing!\n')
    bert_sentences = []
    for sent in tqdm(all_tokenized_sentences):
        bert_sentences.append(bert_embedding.get_embeddings(sent).cpu())
    # 3164840만큼 진행함
    torch.save(bert_sentences,
               'C:/Users/elain/Desktop/glassdoor_aspect_based_sentiment_analysis/sample_data/baseline_bert/english_bert_sentence_embeddings.pt')
     
        
        
    """
     
    test1=bert_embedding.get_embeddings('the best company ever')
    test2=bert_embedding.get_embeddings('toxic work environment')
    test3=bert_embedding.get_embeddings('great place to work for')
    
    
    diff_1 = 1-cosine(test1.cpu(), test2.cpu())
    diff_2 = 1-cosine(test1.cpu(), test3.cpu())
    """