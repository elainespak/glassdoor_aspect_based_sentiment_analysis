# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
pd.set_option('display.max_columns', 50)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class Doc2VecEmbeddings:
    def __init__(self, path, model_path, aspect, tokens, tags):
        self.path = path
        self.model_path = model_path
        self.aspect = aspect
        self.text_type = text_type
        self.tokens = tokens
        self.tags = tags
        
    def get_model(self):
        if os.path.isfile(self.model_path):
            self.doc2vec_model = Doc2Vec.load(self.model_path)
        else:
            df = torch.load(self.path)
            print(f'Original length: {len(df)}')
            if aspect is not None:
                self.df = df[df['aspect']==self.aspect]
                print(f'Aspect {aspect} length: {len(self.df)}')
            else:
                self.df = df
                print('This is a vanilla model')
            doc2vec_corpus = []
            for word, tag in tqdm(zip(self.df[self.tokens], self.df[self.tags])):
                doc2vec_corpus.append(TaggedDocument(word, [tag]))
            self.doc2vec_model = Doc2Vec(doc2vec_corpus)
            self.doc2vec_model.save(self.model_path)
        print(f' The model has {len(self.doc2vec_model.docvecs)} documents')
    
    def get_most_similar_companies(self, gics_path, group, group_number, company_of_interest):
        self.company_gics = torch.load(gics_path)
        print(list(self.company_gics[group].unique()))
        self.companies = list(self.company_gics[self.company_gics[group]==group_number][self.tags].unique())
        print(f' In {group_number} of {group}, \n {self.companies}\n exist')
        
        self.most_similar_companies = []
        for (company, similarity) in self.doc2vec_model.docvecs.most_similar(company_of_interest,topn=len(self.doc2vec_model.docvecs)):
            if company in self.companies:
                self.most_similar_companies.append((company,similarity))
        print(f'\n*** Most similar to {company_of_interest} in {group_number}:')
        print(self.most_similar_companies)
        

if __name__ == '__main__':
    
    # Parameters
    text_type = 'pros'
    aspects = ['CultureAndValues', 'CompensationAndBenefits', 'CareerOpportunities',
               'BusinessOutlook', 'SeniorLeadership', 'WorkLifeBalance']
    path = f'../sample_data/master/{text_type}_12_aspect_labeled.pt'
    gics_path = '../sample_data/master/company_gics.pt'
    tokens = 'trigramSentence'
    tags = 'company'
    
    # Make doc2vec model for aspects
    for aspect in aspects:
        model_path = f'../sample_data/abae/{text_type}/{text_type}_{aspect}_doc2vec_model'
        embeddings = Doc2VecEmbeddings(path, model_path, aspect, tokens, tags)
        embeddings.get_model()
    
    # Make vanilla doc2vec model
    model_path = f'../sample_data/abae/{text_type}/{text_type}_{aspect}_doc2vec_model'
    embeddings = Doc2VecEmbeddings(path, model_path, aspect, tokens, tags)
    embeddings.get_model()
    
    aspect = None
    model_path = f'../sample_data/abae/{text_type}/{text_type}_vanilla_doc2vec_model'
    embeddings = Doc2VecEmbeddings(path, model_path, aspect, tokens, tags)
    embeddings.get_model()

