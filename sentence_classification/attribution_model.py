# -*- coding: utf-8 -*-

# https://captum.ai/tutorials/Bert_SQUAD_Interpret

import nltk
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients
from transformers import BertTokenizer, BertForSequenceClassification


class FineTunedBertForAttribution:
    task = 'BertForSequenceClassification'
    
    def __init__(self, fine_tuned_model, device):
        self.fine_tuned_model = fine_tuned_model
        self.device = device
    
    def retrieve_fine_tuned_bert(self):
        self.model = BertForSequenceClassification.from_pretrained(self.fine_tuned_model)
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        print('Model loaded')
        
        self.tokenizer = BertTokenizer.from_pretrained(self.fine_tuned_model)
        print('Tokenizer loaded')
    
        self.ref_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        print('Special tokens loaded')
    
    def construct_input_ref_pair(self, text):
        self.text = text
        text_ids = self.tokenizer.encode(self.text, add_special_tokens=False)
    
        # construct input token ids
        input_ids = [self.cls_token_id] + text_ids + [self.sep_token_id]
    
        # construct reference token ids 
        ref_input_ids = [self.cls_token_id] + [self.ref_token_id] * len(text_ids) + [self.sep_token_id]
        
        self.input_ids = torch.tensor([input_ids], device=self.device)
        self.ref_input_ids = torch.tensor([ref_input_ids], device=self.device)
        self.sep_id = len(text_ids)
        return self.input_ids, self.ref_input_ids, self.sep_id

    def construct_input_ref_token_type_pair(self):
        seq_len = self.input_ids.size(1)
        self.token_type_ids = torch.tensor([[0 if i <= self.sep_id else 1 for i in range(seq_len)]], device=self.device)
        self.ref_token_type_ids = torch.zeros_like(self.token_type_ids, device=self.device)# * -1
        return self.token_type_ids, self.ref_token_type_ids
    
    def construct_attention_mask(self):
        self.attention_mask = torch.ones_like(self.input_ids)
        return self.attention_mask

    def predict(self, inputs, token_type_ids=None, attention_mask=None):
        return self.model(inputs, token_type_ids=self.token_type_ids, attention_mask=self.attention_mask, )
        
    def predict_label(self):
        self.logits = self.model(self.input_ids, token_type_ids=self.token_type_ids, attention_mask=self.attention_mask)
        self.prediction = torch.argmax(self.logits[0])
        return self.prediction
        #return self.model(self.input_ids, token_type_ids=self.token_type_ids, attention_mask=self.attention_mask, )
    
    def sentclass_pos_forward_func(self, inputs, token_type_ids=None, attention_mask=None):
        pred = self.predict(inputs,
                       token_type_ids=self.token_type_ids,
                       attention_mask=self.attention_mask)
        pred = pred[0]
        return pred.max(1).values
    
    #def sentclass_pos_forward_func(self):
        #logits = self.model(self.input_ids, token_type_ids=None, attention_mask=None)
        #return logits[0].max(1).values
        #pred = self.predict(self.input_ids, token_type_ids=self.token_type_ids, attention_mask=self.attention_mask)
        #pred = pred[0]
        #return pred.max(1).values

    def attribution(self):
        #self.logits = self.model(self.input_ids, token_type_ids=self.token_type_ids, attention_mask=self.attention_mask, )
        #self.prediction = torch.argmax(self.logits[0])
        #self.sentclass_pos_forward_func = self.logits[0].max(1).values
        
        lig = LayerIntegratedGradients(self.sentclass_pos_forward_func, self.model.bert.embeddings)
        attributions_start, self.delta_start = lig.attribute(inputs = self.input_ids,
                                                        baselines = self.ref_input_ids,
                                                        additional_forward_args=(self.token_type_ids, self.attention_mask),
                                                        return_convergence_delta=True)
        attributions_start = attributions_start.sum(dim=-1).squeeze(0)
        self.attributions_start_summary = attributions_start / torch.norm(attributions_start)
        #self.attributions_start_summary = self.attributions_start_summary.detach().tolist()
        return self.attributions_start_summary
    
    def get_top_words(self, ground_truth, custom_stopwords):
        indices = self.input_ids[0].detach().tolist()
        self.ground_truth = ground_truth
        self.all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
        
        if self.prediction == self.ground_truth:
            zipped = zip(self.attributions_start_summary.detach().tolist(), self.all_tokens)
            zipped = [(abs(i[0]), i[1]) for i in zipped if i[1] not in custom_stopwords]
            zipped = sorted(zipped, reverse=True)
            n = len(zipped)//3
            self.top_n = zipped[:n]
            
    def show_results(self):
        print('Review sentence: ', self.text)
        print(f'Predicted Answer: {self.prediction} vs. Gold: {self.ground_truth}')
        for t, a in zip(self.all_tokens, self.attributions_start_summary.detach().tolist()):
            print(t + '\t\t' + str(a))
    
    


def to_sentiment(rating):
    rating = float(rating)
    if rating <= 1.5:
        return 0
    else:
        return 1


def get_highest_attributes():


if __name__ == "__main__":
    
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device, '\n')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        
    # Bring model
    aspect = 'CompensationAndBenefits'
    epochs = 3
    kind = 'all'
    FINE_TUNED_MODEL_NAME = f'./{aspect}_{kind}_epoch{str(epochs)}/'

    aspect = 'CompensationAndBenefits'
    df = torch.load(f'../sample_data/sentence_classification/{aspect}_for_sentence_classification.pt')
    df['labels'] = df['rating'+aspect].apply(to_sentiment)
    df = df[:35000]
    
    custom_stopwords = stopwords.words('english')+[',','.','-', "'", '"', '/']
    attribution_model = FineTunedBertForAttribution(FINE_TUNED_MODEL_NAME, device)
    attribution_model.retrieve_fine_tuned_bert()
    
    #idx = 3
    for idx in range(10):
        text = df.original[idx]
        ground_truth = df.labels[idx]
        attribution_model.construct_input_ref_pair(text)
        attribution_model.construct_input_ref_token_type_pair()
        attribution_model.construct_attention_mask()
        attribution_model.predict_label()
        attribution_model.attribution()
        attribution_model.get_top_words(ground_truth, custom_stopwords)
        #attribution_model.show_results()
        print(attribution_model.top_n)
    
    
    """
    # Doesn't work for some reason..
    # storing couple samples in an array for visualization purposes
    start_position_vis = viz.VisualizationDataRecord(
                            attribution_model.attributions_start_summary,
                            torch.max(attribution_model.logits[0]),
                            attribution_model.prediction,
                            attribution_model.prediction,
                            ground_truth,
                            attribution_model.attributions_start_summary.sum(),       
                            attribution_model.all_tokens,
                            attribution_model.delta_start
                            )
    
    
    print('\033[1m', 'Visualizations For Start Position', '\033[0m')
    from IPython.core.display import display, HTML
    display(HTML(viz.visualize_text([start_position_vis])))
    """

