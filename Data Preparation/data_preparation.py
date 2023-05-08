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
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ner_model import replace_entities
from ner_model import replace_airport
nlp = spacy.load("en_core_web_trf")


# Import Airline Reviews Data
AirlineReviewsData = pd.read_csv('../data/AirlineReviews.csv')

# Remove faulty scraped observations:
# Observations that do not have an airline name in the Airline column:
AirlineNames = pd.read_csv('../data/AirlineReviewCounts.csv')['AirlineName'].tolist()
AirlineReviewsData = AirlineReviewsData[AirlineReviewsData['AirlineName'].isin(AirlineNames)]
# Observations that have N/A in the Review column
AirlineReviewsData = AirlineReviewsData.dropna(subset=['Review'])
# Observations that do not have string data type in Review column
AirlineReviewsData = AirlineReviewsData.loc[AirlineReviewsData['Review'].apply(lambda x: isinstance(x, str))]
# These changes brought us from 128,555 reviews to 74,079 reviews (2023-05-04)

# For test purposes of the code, run it with only the first 10 observations to make sure code is good to be deployed on full data set
AirlineReviewsData10 = AirlineReviewsData.head(10)

# Text pre-processing tools
# stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')


# Preprocess each document (i.e. each review)
def preprocess_reviews(review):
    # convert non-english characters to ASCII
    review = unidecode(review)

    # remove punctuation/symbols/non-space characters/etc
    review = re.sub(r"[^\w\s]", '', review)

    # replace common words not categorized by NER, i.e. "MIA" becomes "_airport_". Also removes stop words.
    review = replace_airport([review])

    # NER classify entities and substitute them by a common word
    document = nlp(review)
    review = replace_entities(document)

    # lowercase
    review = review.lower()

    # remove words that only appear once.
    # frequency = defaultdict(int)
    # for word in review:
    #    frequency[word] += 1
    # review = [word for word in review if frequency[word] > 1]
    # Many reviews are less than 100 words, therefore this step is intentionally left out.

    # POS (tag words with grammar and only keep adjectives and nouns). This step gets rid of _, will need to keep as is.
    doc = nlp(review)
    review = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]

    # lemmatize reviews
    review = [lemmatizer.lemmatize(word) for word in review]

    # tokenize each document, step not required as the POS tagging does this already
    # review = tokenizer.tokenize(review)

    return review


AirlineReviewsData10.loc[:, 'Pre-processed Reviews'] = AirlineReviewsData10['Review'].apply(preprocess_reviews)

print(AirlineReviewsData10['Pre-processed Reviews'])





