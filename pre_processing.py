# -*- coding: utf-8 -*-
# Libs import
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import spacy
import pandas as pd
from tqdm import tqdm
STOPWORDS = set(stopwords.words('german'))
import numpy as np
import unicodedata
import unidecode
import spacy.cli
import gensim
spacy.cli.download("de_core_news_sm")
nlp = spacy.load("de_core_news_sm")
import math
import os

# Functions
def cleaning_pipeline(text):
    """
    clean the data before processing
    """

    # remove stopwords from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)

    # removes any words composed of less than 3
    text = ' '.join(word for word in text.split() if (len(word) >= 3))

    # removes accents
    text = ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))

    text = text.lower()
    text = re.sub(r"[_/(){}\[\]\|@,;]", " ", text)
    text = re.sub(r"[^0-9a-z #+_]", "", text)
    text = re.sub(r"[0-9]+", " ", text)
    text = re.sub(r"[\d+]", "", text)

    # Lemm
    text = ' '.join(word.lemma_.lower() for word in nlp(text))

    # Stemming the words
    text = ' '.join(stemmer.stem(word) for word in text.split())

    return text


# Read data into papers
df_programmes = pd.read_csv('programmes_processed.csv', sep=';', encoding='utf-8')
df_articles = pd.read_csv('articles_processed_final.csv', sep=';', encoding='utf-8')


"""Processing programmes"""

corpus_programmes = []

for i in range(len(df_programmes['programmes_processed'])):
    doc = nlp(cleaning_pipeline(df_programmes['programmes_processed'][i]))
    for token in doc:
        if token.lemma_.lower():
            corpus_programmes.append(token.lemma_.lower())
        else:
            corpus_programmes.append(token)
            
            
programmes_df = pd.DataFrame(corpus_programmes,columns=['programmes'])


"""Processing articles"""

corpus_articles = []

for i in range(len(df_programmes['articles_processed'])):
    doc = nlp(cleaning_pipeline(df_programmes['articles_processed'][i]))
    for token in doc:
        if token.lemma_.lower():
            corpus_articles.append(token.lemma_.lower())
        else:
            corpus_articles.append(token)
            
            
articles_df = pd.DataFrame(corpus_articles,columns=['programmes'])