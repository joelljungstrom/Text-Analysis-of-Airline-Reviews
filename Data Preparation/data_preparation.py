import pandas as pd
import sys
import os
import numpy as np
import re
import spacy
import nltk
import swifter
import time
import statistics
import dask.dataframe as dd
import multiprocessing
import matplotlib.pyplot as plt
from dask import delayed, compute
from multiprocessing import Pool
from unidecode import unidecode
from collections import defaultdict
from gensim import corpora
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from ner_model import replace_entities
from ner_model import replace_airport
nlp = spacy.load("en_core_web_md")


# Import Airline Reviews Data
AirlineReviewsData = pd.read_csv('../Data/AirlineReviews.csv')

# Remove faulty scraped observations:
# Observations that do not have an airline name in the Airline column:
AirlineSlug = pd.read_csv('../Data/AirlineReviewCounts.csv', usecols=['Slug'])['Slug'].tolist()
AirlineReviewsData = AirlineReviewsData[AirlineReviewsData['Slug'].isin(AirlineSlug)]
# Observations that have N/A in the Review column
AirlineReviewsData = AirlineReviewsData.dropna(subset=['Review'])
# Observations that do not have string data type in Review column
AirlineReviewsData = AirlineReviewsData.loc[AirlineReviewsData['Review'].apply(lambda x: isinstance(x, str))]
AirlineReviewsData.to_csv('../Data/AirlineReviewsData.csv', index=False)
# These changes brought us from 129,455 reviews to 128,631 reviews (2023-05-09)

# For test purposes of the code, run it with only the first 100 observations to make sure code is good to be deployed on full data set
# AirlineReviewsData100 = AirlineReviewsData.head(100)


# Preprocess each document (i.e. each review)
def preprocess_reviews(review):
    # convert non-english characters to ASCII
    review = unidecode(review)

    # remove punctuation/symbols/non-space characters/etc
    review = re.sub(r"[^\w\s]", '', review)

    # replace common words not categorized by NER, i.e. "MIA" becomes "_airport_". Also removes stop words.
    review = replace_airport([review])

    # NER classify entities and substitute them by a common word
    doc = nlp(review)
    review = replace_entities(doc).lower()

    # lowercase
    review = review.lower()

    # remove words that only appear once.
    # frequency = defaultdict(int)
    # for word in review:
    #    frequency[word] += 1
    # review = [word for word in review if frequency[word] > 1]
    # Many reviews are less than 100 words, therefore this step is intentionally left out.

    # POS & lemmatize reviews (only keep adjectives and nouns). This step gets rid of _, will need to keep as is.
    review = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]

    return review


# Attempt to use parallel processing
# if __name__ == '__main__':
#     start_time = time.time()
#
#     from ner_model import replace_entities
#     from ner_model import replace_airport
#     AirlineSlug = pd.read_csv('../Data/AirlineReviewCounts.csv', usecols=['Slug'])['Slug'].tolist()
#     AirlineReviewsData = AirlineReviewsData[AirlineReviewsData['Slug'].isin(AirlineSlug)]
#     # Observations that have N/A in the Review column
#     AirlineReviewsData = AirlineReviewsData.dropna(subset=['Review'])
#     # Observations that do not have string data type in Review column
#     AirlineReviewsData = AirlineReviewsData.loc[AirlineReviewsData['Review'].apply(lambda x: isinstance(x, str))]
#     # These changes brought us from 129,455 reviews to 128,631 reviews (2023-05-09)
#
#     AirlineReviewsData1000 = AirlineReviewsData.head(1000)
#
#     pool = Pool(processes=2)
#
#     AirlineReviewsData1000['Pre-processed Reviews'] = AirlineReviewsData1000['Review'].apply(preprocess_reviews)
#
#     end_time = time.time()
#
#     print(f"Elapsed time: {end_time - start_time:.2f} seconds")

start_time = time.time()
# split file up into multiple frames to batch the work
dfs = np.array_split(AirlineReviewsData, len(AirlineReviewsData) // 5000 + 1)
for i in range(len(dfs)):
    dfs[i]['Pre-processed Reviews'] = dfs[i]['Review'].apply(preprocess_reviews)
    if i == 0:
        dfs[i].to_csv('../Data/preprocessed_reviews.csv', index=False)
        print(i * len(dfs[i]))
    else:
        dfs[i].to_csv('../Data/preprocessed_reviews.csv', mode='a', index=False, header=False)
        print(i * len(dfs[i]))

end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")

# replace missing "Overall Score" scores (4,330 occurrences, 3.37%)
rating_replace_value = np.nanmean(AirlineReviewsData["OverallScore"]).round() # 4.5... rounded to 5
AirlineReviewsData["OverallScore"].fillna(rating_replace_value, inplace=True)
