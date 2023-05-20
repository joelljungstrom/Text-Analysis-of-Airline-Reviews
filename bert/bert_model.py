import gensim.models
import pandas as pd
import sys
import os
import numpy as np
import re
import spacy
import csv
import nltk
import torch
import transformers
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipelines
from unidecode import unidecode
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ner_model import replace_entities
from ner_model import replace_airport
from multiprocessing import set_start_method
from multiprocessing import Process
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
nlp = spacy.load("en_core_web_sm")


# import Processed Data
AllData = pd.read_csv('../Data/preprocessed_reviews.csv')
Reviews = AllData['Pre-processed Reviews'].tolist()
Reviews = [eval(review) for review in Reviews]
Reviews = [' '.join(review) for review in Reviews]

dictionary = corpora.Dictionary([review.split() for review in Reviews])
corpus = [dictionary.doc2bow(review.split()) for review in Reviews]

bert_model = 'textattack/bert-base-uncased-imdb'
bert = AutoModelForSequenceClassification.from_pretrained(bert_model)
tokenizer = AutoTokenizer.from_pretrained(bert_model)





