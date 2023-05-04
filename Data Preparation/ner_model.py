import pandas as pd
import csv
import os
import numpy as np
import re
import spacy
import nltk
from spacy.tokens import Doc, Span, Token
from unidecode import unidecode
from collections import defaultdict
from gensim import corpora
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import Airports/Cities/Countries data
file_path = '../data/airports_cities_countries.csv' # https://github.com/jpatokal/openflights/blob/master/data/airports.dat
AirInfo = pd.read_csv(file_path, sep = ",")

nlp = spacy.load("en_core_web_trf")

# Create a function to replace common words by an entity using SpaCy NER
def replace_entities(document):
    new_document = []
    for token in document:
        if token.ent_type_ == "GPE": # countries/cities/states
            new_document.append("_location_")
        elif token.ent_type_ == "LOC": # mountains/water bodies
            new_document.append("_location_")
        elif token.ent_type_ == "FAC": # airports
            new_document.append("_airport_")
        elif token.ent_type_ == "ORG":
            new_document.append("_organization_")
        elif token.ent_type_ == "PERSON":
            new_document.append("_person_")
        elif token.ent_type_ == "PRODUCT":
            new_document.append("_product_")
        elif token.ent_type_ == "WORK_OF_ART":
            new_document.append("_creative_")
        elif token.ent_type_ == "LAW":
            new_document.append("_law_")
        elif token.ent_type_ == "LANGUAGE":
            new_document.append("_lang_")
        elif token.ent_type_ == "DATE":
            new_document.append("_date_")
        elif token.ent_type_ == "TIME":
            new_document.append("_time_")
        elif token.ent_type_ == "PERCENT":
            new_document.append("_percent_")
        elif token.ent_type_ == "MONEY":
            new_document.append("_price_")
        elif token.ent_type_ == "QUANTITY":
            new_document.append("_weight_")
        elif token.ent_type_ == "CARDINAL":
            new_document.append("_number_")
        elif token.ent_type_ == "ORDINAL":
            new_document.append("_leg_")
        elif token.ent_type_ == "NORP":
            new_document.append("_nationality_")
        else:
            new_document.append(token.text)
    return ' '.join(new_document)

# Can we train this to recognize more entities? i.e. "COR-EZE" get's recognized as route?











