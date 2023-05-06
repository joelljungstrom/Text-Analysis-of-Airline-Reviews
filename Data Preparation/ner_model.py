import random
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
from spacy.training import Example

# Import Airports/Cities/Countries data
AirInfo = pd.read_csv('../data/airports_cities_countries.csv', sep = ",") # https://github.com/jpatokal/openflights/blob/master/data/airports.dat
AirInfo = AirInfo[['airport_name','code']]

# Import Route from review csv
RouteInfo = pd.read_csv('../data/airlineReviews.csv')
RouteInfo = RouteInfo['Route']
RouteInfo = RouteInfo.to_frame()
RouteInfo = RouteInfo.dropna()

nlp = spacy.load("en_core_web_trf")

# PC isn't capable of training custom fields to the Spacy model.
# Blank English model
# nlp_blank = spacy.blank("en")

# Labels for entities to recognize
# labels = ["ROUTE", "AIRPORT"]

# train_data = []
# for _, row in RouteInfo.iterrows():
#     text = row["Route"]
#     entities = [(text.index(row["Route"]), text.index(row["Route"]) + len(row["Route"]), "ROUTE")]
#     example = Example.from_dict(nlp_blank.make_doc(text), {"entities": entities})
#     train_data.append(example)

# for _, row in AirInfo.iterrows():
#    text = row["airport_name"]
#    entities = [(text.index(row["airport_name"]), text.index(row["airport_name"]) + len(row["airport_name"]), "AIRPORT")]
#    example = Example.from_dict(nlp_blank.make_doc(text), {"entities": entities})
#    train_data.append(example)

#for _, row in AirInfo.iterrows():
#    text = row["code"]
#    entities = [(text.index(row["code"]), text.index(row["code"]) + len(row["code"]), "AIRPORT")]
#    example = Example.from_dict(nlp_blank.make_doc(text), {"entities": entities})
#    train_data.append(example)

# nlp_blank.begin_training()
# for i in range (10):
#     random.shuffle(train_data)
#     losses = {}
#     nlp.update(train_data, losses = losses, drop = 0.2)
#     print("Losses:", losses)


# Create a function to replace common words by an entity using SpaCy NER
def replace_entities(document):
    new_document = []
    prev_ent_type = None
    for token in document:
        if token.ent_type_ == prev_ent_type:
            new_document[-1] += "" + token.text
        else:
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
            prev_ent_type = token.ent_type_
    return new_document


airport_pattern = re.compile(r'\b(' + '|'.join(AirInfo['code'] + AirInfo['airport_name']) + r')\b')
route_pattern = re.compile(r'\b(' + '|'.join([code + r' - ' + code for code in AirInfo['code']]) + r')\b')


def replace_airport(document):
    new_document = []
    for i in range(len(document)):
        document[i] = route_pattern.sub("_route_", document[i])
        document[i] = airport_pattern.sub("_airport_", document[i])
    return document











