# -*- coding: utf-8 -*-

# https://captum.ai/tutorials/Bert_SQUAD_Interpret


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients
from transformers import BertTokenizer, BertForSequenceClassification


def predict(inputs, token_type_ids=None, attention_mask=None):
    return model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask, )
    
def sentclass_pos_forward_func(inputs, token_type_ids=None, attention_mask=None, position=0):
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values

def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_bert_sub_embedding(input_ids, ref_input_ids,
                                   token_type_ids, ref_token_type_ids,
                                   ref_position_ids):
    input_embeddings = interpretable_embedding1.indices_to_embeddings(input_ids)
    ref_input_embeddings = interpretable_embedding1.indices_to_embeddings(ref_input_ids)

    input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(token_type_ids)
    ref_input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(ref_token_type_ids)

    return (input_embeddings, ref_input_embeddings), \
           (input_embeddings_token_type, ref_input_embeddings_token_type)
    
def construct_whole_bert_embeddings(input_ids, ref_input_ids, \
                                    token_type_ids=None, ref_token_type_ids=None, \
                                    ref_position_ids=None):
    input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids, token_type_ids=token_type_ids)
    ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids, token_type_ids=token_type_ids)
    
    return input_embeddings, ref_input_embeddings

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def to_sentiment(rating):
    rating = float(rating)
    if rating <= 2:
        return 0
    else:
        return 1


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
    FINE_TUNED_MODEL_NAME = f'C:/Users/elain/Desktop/glassdoor_aspect_based_sentiment_analysis/sentence_classification/toy_{aspect}/'
    
    model = BertForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_NAME)
    model.to(device)
    model.eval()
    model.zero_grad()
    
    tokenizer = BertTokenizer.from_pretrained(FINE_TUNED_MODEL_NAME)
    
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
    
    
    aspect = 'CompensationAndBenefits'
    df = torch.load(f'C:/Users/elain/Desktop/glassdoor_aspect_based_sentiment_analysis/sample_data/sentence_classification/{aspect}_for_sentence_classification.pt')
    df['labels'] = df['rating'+aspect].apply(to_sentiment)
    df = df[:10000]
    
    idx = 4070
    text = df.original[idx]
    ground_truth = df.labels[idx]
    
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    attention_mask = construct_attention_mask(input_ids)
    
    indices = input_ids[0].detach().tolist() # take off from cuda to cpu
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    logits = predict(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    
    lig = LayerIntegratedGradients(sentclass_pos_forward_func, model.bert.embeddings)
    
    attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                      baselines=ref_input_ids,
                                      additional_forward_args=(token_type_ids, attention_mask),
                                      return_convergence_delta=True)
    
    attributions_start_sum = summarize_attributions(attributions_start)
    
    print('Review sentence: ', text)
    print(f'Predicted Answer: {torch.argmax(logits[0])}')
    print(f'Gold: {ground_truth}')
    print(all_tokens)
    print(attributions_start_sum)

"""
# storing couple samples in an array for visualization purposes
start_position_vis = viz.VisualizationDataRecord(
                        attributions_start_sum,
                        torch.max(torch.softmax(logits[0], dim=1)),
                        torch.argmax(logits[0]),
                        torch.argmax(logits[0]),
                        str(ground_truth),
                        attributions_start_sum.sum(),       
                        all_tokens,
                        delta_start)


print('\033[1m', 'Visualizations For Start Position', '\033[0m')
viz.visualize_text([start_position_vis])
"""

