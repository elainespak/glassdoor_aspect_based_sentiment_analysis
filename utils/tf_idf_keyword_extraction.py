# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
from collections import Counter
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_columns', 50)


def get_most_common_trigrams(words, n=100):
    keywords = Counter(w for w in words if len(w.split('_')) >= 3).most_common(n)
    return [word for word, freq in keywords]


def calculate_tf_idf(df):
    vectorizer = TfidfVectorizer()
    words = [' '.join(w) for w in df['words']]
    vectors = vectorizer.fit_transform(words)
    #print('Done vectorizing for tf-idf')
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    result = pd.DataFrame(denselist, columns=feature_names)
    return result


if __name__ == "__main__":
     
    