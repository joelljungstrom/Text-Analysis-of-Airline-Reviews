import pandas as pd
import os
import numpy as np
import re
import spacy
import nltk
from unidecode import unidecode
from collections import defaultdict
from gensim import corpora
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



# Import Airline Reviews Data
AirlineReviewsData = pd.read_csv('/Users/joelljungstrom/PycharmProjects/AirlineReviewCrawler/AirlineReviews/AirlineReviews/spiders/AirlineReviews.csv')

# Remove faulty scraped observations:
# Observations that do not have an airline name in the Airline column:
AirlineNames = pd.read_csv('/Users/joelljungstrom/PycharmProjects/AirlineReviewOverview/AirlineReviewCounts/AirlineReviewCounts/spiders/AirlineReviewCounts.csv')['AirlineName'].tolist()
AirlineReviewsData = AirlineReviewsData[AirlineReviewsData['AirlineName'].isin(AirlineNames)]
# Observations that have N/A in the Review column
AirlineReviewsData = AirlineReviewsData.dropna(subset=['Review'])
# Observations that do not have string data type in Review column
AirlineReviewsData = AirlineReviewsData.loc[AirlineReviewsData['Review'].apply(lambda x: isinstance(x, str))]
# These changes brought us from 128,555 reviews to 74,079 reviews (2023-05-04)


# Text pre-processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
ner = spacy.load("en_core_web_sm")


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


# Preprocess each document (i.e. each review)
def preprocess_reviews(review):
    # lowercase
    review = review.lower()

    # remove stop words
    review = [word for word in review.split() if word not in stop_words]

    # convert non-english characters to ASCII
    review = unidecode(review)

    # remove words that only appear once
    frequency = defaultdict(int)
    for word in review:
        frequency[word] += 1
    review = [word for word in review if frequency[word] > 1]

    # NER (classify entities and substitute them by a common word, i.e. "MIA" becomes "_airport_"
    document = ner(' '.join(review))
    review = replace_entities(document)

    # lemmatize reviews
    review = [lemmatizer.lemmatize(word) for word in review]

    # remove punctuation/symbols/non-space characters/etc
    review = re.sub(r"[^\w\s]", '', review)

    # tokenize each document
    review = tokenizer.tokenize(' '.join(review))

    return review


AirlineReviewsData['Pre-processed Review'] = AirlineReviewsData['Review'].apply(preprocess_reviews)


# Left to do:
# POS tagging (tag words with grammar)





