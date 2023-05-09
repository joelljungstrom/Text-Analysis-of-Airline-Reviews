import gensim.models
import pandas as pd
import sys
import os
import numpy as np
import re
import spacy
import nltk
from unidecode import unidecode
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim.models import LdaModel
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ner_model import replace_entities
from ner_model import replace_airport
nlp = spacy.load("en_core_web_trf")

# Import Processed Data
AllData = pd.read_csv('../Data/preprocessed_reviews.csv')
Reviews = AllData['Pre-processed Reviews']

# Tokenize each review
tokenizer = RegexpTokenizer(r'\w+')
review = [tokenizer.tokenize(review) for review in Reviews]

dictionary = corpora.Dictionary(review)
corpus = [dictionary.doc2bow(review) for review in review] # doc2bow counts the number of occurrences of each distinct word

num_topics = 9
passes = 10

first_model = LdaModel(corpus = corpus,
                       id2word = dictionary,
                       num_topics = num_topics,
                       passes = passes)

topics = first_model.show_topics()
for topic in topics:
    print(topic)