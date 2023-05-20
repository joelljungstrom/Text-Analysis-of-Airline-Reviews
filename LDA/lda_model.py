import gensim.models
import pandas as pd
import sys
import os
import numpy as np
import re
import spacy
import csv
import nltk
import matplotlib.pyplot as plt
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

dictionary = corpora.Dictionary([review.split() for review in Reviews]) # corpora.Dictionary maps each unique word to an integer id
corpus = [dictionary.doc2bow(review.split()) for review in Reviews] # doc2bow counts the number of occurrences of each distinct word


# k-fold cross validation on topics 2-20, calculate coherence scores and save models and scores
def lda_models(corpus, dictionary, num_topics_range, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    models = []
    coherence_scores = []

    for train_index, test_index in kf.split(corpus):
        train_corpus = [corpus[i] for i in train_index]
        test_corpus = [corpus[i] for i in test_index]

        for num_topics in num_topics_range:

            model = LdaModel(corpus=train_corpus,
                             id2word=dictionary,
                             num_topics=num_topics)
            coherence_model = CoherenceModel(model=model,
                                             texts=Reviews,
                                             dictionary=dictionary,
                                             corpus=test_corpus,
                                             coherence='c_v')
            coherence_score = coherence_model.get_coherence()

            models.append(model)
            coherence_scores.append(coherence_score)

    return models, coherence_scores


if __name__ == '__main__':
    num_topics_range = range(2, 20)
    models, coherence_scores = lda_models(corpus, dictionary, num_topics_range)


# calculate average coherence score per num_topics
coherence_scores_avg = {}
for model, coherence_score in zip(models, coherence_scores):
    num_topics = model.num_topics-1
    if num_topics not in coherence_scores_avg:
        coherence_scores_avg[num_topics] = []
    coherence_scores_avg[num_topics].append(coherence_score)
coherence_scores_avg = {
    num_topics: sum(scores) / len(scores) for num_topics, scores in coherence_scores_avg.items()
}

# combine models with the same num_topics
combined_models = {}
for model, coherence_score in zip(models, coherence_scores):
    num_topics = model.num_topics-1
    if num_topics not in combined_models:
        combined_models[num_topics] = model
    else:
        combined_models[num_topics].update(model)

# save models to repository
lda_saved_models = "lda_saved_models"
for num_topics, model_list in combined_models.items():
    if isinstance(model_list, int):
        continue
    for i, model in enumerate(model_list):
        filename = f"lda_model_{num_topics}_{i+1}_topics"  # include an index to differentiate multiple models with the same number of topics
        save_path = os.path.join(lda_saved_models, filename)
        if isinstance(model, int):
            continue
        model.save(save_path)

# save coherence scores to csv file
lda_saved_coherence_scores = "lda_saved_coherence_scores"
os.makedirs(lda_saved_coherence_scores, exist_ok=True)
filename = "coherence_scores.csv"
save_path = os.path.join(lda_saved_coherence_scores, filename)
with open(save_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Num_Topics', 'Coherence_Score'])
    for num_topics, coherence_score in coherence_scores_avg.items():
        if isinstance(coherence_score, int):
            continue
        writer.writerow([num_topics, coherence_score])


# load final LDA model with the highest coherence score
final_model = LdaModel.load('lda_saved_models/lda_model_18_topics')
# assign identified words for each topic to "topics"
topics = final_model.show_topics(18)


# rename the topics based on identified dimension
topic_labels = {
    0: "Travel Delays",
    1: "Luggage Handling",
    2: "Good Service",
    3: "Efficiency",
    4: "Price",
    5: "In-Flight Beverage",
    6: "Customer Service",
    7: "Carry-on",
    8: "Frequent Flyer Program",
    9: "Charges",
    10: "In-Flight Experience",
    11: "Cabin Crew",
    12: "Seat Comfort",
    13: "Family-Friendly",
    14: "Business Class",
    15: "Boarding Process",
    16: "Airline Experience",
    17: "Ground Experience"
}


topics_with_labels = []
for topic in topics:
    topic_id, words_weights = topic[0], topic[1:]
    topic_label = topic_labels.get(topic_id)
    topic_with_label = (topic_label,) + words_weights
    topics_with_labels.append(topic_with_label)


# get distribution of topic over entire corpus
topic_distribution = []
for document in corpus:
    doc_topics = final_model.get_document_topics(document, minimum_probability=0)
    topic_distribution.append(doc_topics)


# attain average probability and standard deviation of topic occurrence in corpus
topic_weights = [0] * num_topics
topic_std = []
for doc_topics in topic_distribution:
    for topics_with_labels, topic_weight in doc_topics:
        topic_weights[topics_with_labels] += topic_weight
topic_distribution_average = [weight / 128631 for weight in topic_weights] # divide by number of documents to gain a comparable average probability among the dimensions
for topic_idx in range(num_topics):
    topic_std_weight = np.array([doc_topics[topic_idx][1] for doc_topics in topic_distribution], dtype=np.float32)
    topic_std.append(np.std(topic_std_weight))
topic_std_avg_weight = sorted(zip(topic_distribution_average, topic_std), key=lambda x: x[0], reverse=True)
topic_std_avg_weight = pd.DataFrame(topic_std_avg_weight, columns=['Avg Weight', 'Std'])


# visualize
x = np.arange(0, len(topic_labels))
average = topic_std_avg_weight['Avg Weight']
std = topic_std_avg_weight['Std']

plt.errorbar(x, average, yerr=std, fmt='o', capsize=5)
plt.xlabel('X')
plt.ylabel('Average weight')
plt.title('Average with Standard Deviation')
plt.show()
