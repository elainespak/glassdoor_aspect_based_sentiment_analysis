# -*- coding: utf-8 -*-


import torch
import logging
from tqdm import tqdm
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel


class BERTEmbedding:

    def __init__(self, device, model_type):
         self.device = device
         self.model_type = model_type
         self.model, self.tokenizer = self.retrieve_bert()
         
         # Put the model in "evaluation" mode, meaning feed-forward operation.
         self.model.eval()

    def retrieve_bert(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_type,
                                                  output_hidden_states = True)
        model = BertModel.from_pretrained(self.model_type,
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
        return model.to(self.device), tokenizer

    def get_embeddings(self, sent):
        marked_sent = '[CLS] ' + sent + ' [SEP]'
        
        # Split the sentence into tokens.
        tokenized_sent = self.tokenizer.tokenize(marked_sent)

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
     path = 'C:/Users/elain/Desktop/glassdoor_aspect_based_sentiment_analysis/sample_data/abae/glassdoor/test.txt'
     sentence_bert = []
     with open(path, 'r') as f:
         f = f.readlines()
         for sent in tqdm(f):
             sentence_bert.append(bert_embedding.get_embeddings(sent))
     torch.save('C:/Users/elain/Desktop/glassdoor_aspect_based_sentiment_analysis/sample_data/abae/glassdoor/test.pt')
     
     
        
     """
     test1=bert_embedding.get_embeddings('the best company ever')
     test2=bert_embedding.get_embeddings('toxic work environment')
     test3=bert_embedding.get_embeddings('great place to work for')
     
     
     diff_1 = 1-cosine(test1.cpu(), test2.cpu())
     diff_2 = 1-cosine(test1.cpu(), test3.cpu())
     """