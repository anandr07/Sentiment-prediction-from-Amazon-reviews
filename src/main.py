#!/usr/bin/env python
# coding: utf-8

#%%[markdown]
# # Sentiment prediction from Amazon reviews

# ## About DataSet

# This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.
# 
# Contents
# 
# database.csv: Contains the table 'Reviews'
# 
# Data includes:
# 
# Reviews from Oct 1999 - Oct 2012 568,454 reviews 256,059 users 74,258 products 260 users with > 50 reviews


#%%
import sys
import os

# Getting the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adding the parent directory to the Python path
sys.path.append(os.path.dirname(current_dir))


#%%
#Importing Libraries

import sys 
import sqlite3
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import warnings
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle
import gensim
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import clean_text, preprocess_text, sentence_to_words
from ml_algorithms.KNN import KNN_train_simple_cv
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer


#%%
raw_data = pd.read_csv("C:\Anand\Projects_GWU\Sentiment-prediction-from-Amazon-reviews\data\Reviews.csv")

#%%
raw_data.head(10)

#%%
print(raw_data["Text"].head(10))

#%%
raw_data.shape

#%%

# Drop rows with rating/score as 3.
value_to_drop = 3

# Drop rows where 'Score' has value 3.
raw_data = raw_data[raw_data['Score'] != value_to_drop]


#%%
# After dropping row with score 3
print(raw_data.shape)

#Unique values in Score column must be 1/2/4/5.
print(raw_data.Score.unique())


#%%

# Giving 4&5 as Positive and 1&2 as Negative Rating 
def assign_values(value):
    if value < 3:
        return 'Negative'
    else:
        return 'Positive'

raw_data['Review'] = raw_data['Score'].apply(assign_values)


#%%
raw_data.head(5)


#%%
# Checking for duplicate Reviews 
boolean = not raw_data["Text"].is_unique      
boolean = raw_data['Text'].duplicated().any()
print(boolean)


#%%
# Drop duplicated Reviews
raw_data = raw_data.drop_duplicates(subset='Text', keep='first')

# Check the shape
print(raw_data.shape)


#%%
# Check if HelpfulnessNumerator is less than HelpfulnessDenominator, If so then drop those rows
raw_data=raw_data[raw_data.HelpfulnessNumerator<=raw_data.HelpfulnessDenominator]
print(raw_data.shape)

# The observations in the dataset dropped from 568454 to 363834 as there were a lot of Duplicate Reviews and Number of people who found review helpful cannot be greater than number of people who viewed the review. These rows were dropped

# # Check proportions of categories in output label:
# raw_data['Review'].value_counts()

#%%
# Apply clean_text function to clean the text column.
raw_data['Clean_Text'] = raw_data['Text'].apply(lambda x: clean_text(x))


#%%
print(raw_data["Text"].head(10))


#%%
# Apply text preprocessing to the 'Text' column from data-preprocessing.py file 
raw_data['Clean_Text'] = raw_data['Clean_Text'].apply(preprocess_text)


#%%
# Comparing the Original Text and the processed text
print(raw_data["Text"][1])

print("\n After Processing of Text \n")

print(raw_data["Clean_Text"][1])


#%%
# Converting sentence into words using sentence_to_words function from the data_preprocessing file
list_of_words_in_sentance = sentence_to_words(raw_data, 'Clean_Text')


#%%
raw_data["Clean_Text"].iloc[0]


#%%
print(raw_data['Clean_Text'])


#%%
print(f"Sentence cleaned: {raw_data['Clean_Text'].values[0]}")
print(f"Words in cleaned sentence{list_of_words_in_sentance[0]}")

#%%
#%%[markdown]
# Time Based Splitting 

#%%
final_reviews = raw_data.sort_values('Time', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
final_reviews.shape

#%%
# Label Encoding Reviews Column
label_encoder = LabelEncoder()

# Fit and transform the "Review" column
raw_data['Review'] = label_encoder.fit_transform(raw_data['Review'])

#%%[markdown]
# Train and Test Split

#%%
# Splitting data into train, Train and Test 
X = raw_data['Clean_Text']
Y = raw_data['Review']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=123)

print('X_train, Y_train', X_train.shape, Y_train.shape)
print('X_test, Y_test', X_test.shape, Y_test.shape)

#%%[markdown]
## Bag of Words 

#%%
# Bag Of Words
Count_vectorizer = CountVectorizer()
X_train_bow = Count_vectorizer.fit_transform(X_train.values)
print(f"Shape of dataset after converting into BOW is {X_train_bow.get_shape()}")
X_test_bow = Count_vectorizer.transform(X_test.values)  # Use transform instead of fit_transform
print(f"Shape of dataset after converting into BOW is {X_test_bow.get_shape()}")

#%%
# Normalize BOW Train and Test Data
X_train_bow=preprocessing.normalize(X_train_bow)
X_test_bow=preprocessing.normalize(X_test_bow)
print("The shape of out text BOW vectorizer ",X_train_bow.get_shape())
print("Test Data Size: ",X_test_bow.shape)

#%%[markdown]
## Uni, Bi and Tri Grams

#%%
Count_vectorizer_n_grams = CountVectorizer(ngram_range=(1,3) ) 
X_train_n_grams = Count_vectorizer_n_grams.fit_transform(X_train.values)
print("Shape of dataset after converting into uni, bi and tri-grams is ",X_train_n_grams.get_shape())
X_test_n_grams = Count_vectorizer_n_grams.transform(X_test.values)
print("Shape of dataset after converting into uni, bi and tri-grams is ",X_test_n_grams.get_shape())

#%%[markdown]
## Tf-Idf Vectorization

#%%
# tf-idf Vectorizer
tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train_tf_idf_vectorizer = tf_idf_vectorizer.fit_transform(X_train.values)
X_test_tf_idf_vectorizer = tf_idf_vectorizer.transform(X_test.values)

print("Shape of dataset after converting into tf-idf is ", X_train_tf_idf_vectorizer.get_shape())
print("Shape of dataset after converting into tf-idf is ", X_test_tf_idf_vectorizer.get_shape())

#%%
# Normalize Tf-Idf Train and Test Data
X_train_tfidf=preprocessing.normalize(X_train_tf_idf_vectorizer)
X_test_tfidf=preprocessing.normalize(X_test_tf_idf_vectorizer)
print("The shape of out text BOW vectorizer ",X_train_tfidf.get_shape())
print("Test Data Size: ",X_test_tfidf.shape)

#%%[markdown]
## word2vec Model
# Making word2vec model using our data set and the same model will be used further.

#%%
# Training word2vec model on our own data.
w2v_model=gensim.models.Word2Vec(list_of_words_in_sentance,min_count=5, workers=4) 


#%%
# Saving the vocabolary of words in our trained word2vec model
w2v_vocab = list(w2v_model.wv.key_to_index)


#%%
# Get the top 10 words most similar words to "quality"
w2v_model.wv.most_similar('good')


#%%
raw_data.shape[0]

#%%[markdown]
## Average word2vec

#%%
sent_vectors_avg_word2vec = []; # The avg-w2v for each sentence/review is stored in this list
vector_size = len(w2v_model.wv['good']) 

for sent in tqdm(list_of_words_in_sentance): # Iterating over each review/sentence
    sent_vec = np.zeros(vector_size) 
    cnt_words =0; 
    for word in sent: # Iterating over each word in a review/sentence
        if word in w2v_vocab:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors_avg_word2vec.append(sent_vec)
print(len(sent_vectors_avg_word2vec))

#%%
X_train_avg_wor2vec, X_test_avg_wor2vec, Y_train_avg_wor2vec, Y_test_avg_wor2vec = train_test_split(sent_vectors_avg_word2vec,Y, test_size=.20, random_state=0)
X_train_avg_wor2vec=preprocessing.normalize(X_train_avg_wor2vec)
X_test_avg_wor2vec=preprocessing.normalize(X_test_avg_wor2vec)
print(X_train_avg_wor2vec.shape)
print(X_test_avg_wor2vec.shape)

#%%[markdown]
## Tf-Idf Wword2vec 

#%%
tfidf_model = TfidfVectorizer()
tf_idf_matrix = tfidf_model.fit_transform(raw_data['Clean_Text'].values)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))

# TF-IDF weighted Word2Vec
tfidf_feat = tfidf_model.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in tqdm(list_of_words_in_sentance): # for each review/sentence 
    sent_vec = np.zeros(vector_size) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_vocab:
            vec = w2v_model.wv[word]
            #  tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]
            # to reduce the computation we are 
            # dictionary[word] = idf value of word in whole courpus
            # sent.count(word) = tf valeus of word in this review
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1

X_train_tfidf_word2vec, X_test_tfidf_word2vec, Y_train_tfidf_wor2vec, Y_test_tfidf_wor2vec = train_test_split(tfidf_sent_vectors, Y, test_size=0.20,random_state=0)
X_train_tfidf_word2vec=preprocessing.normalize(X_train_tfidf_word2vec)
X_test_tfidf_word2vec=preprocessing.normalize(X_test_tfidf_word2vec)
print(X_train_tfidf_word2vec.shape)
print(X_test_tfidf_word2vec.shape)

#%%
auc_score_bow_test, accuracy_bow_test = KNN_train_simple_cv(X_train_bow, Y_train, X_test_bow, Y_test)

#%%[markdown]
# Best K found after cross-validation is 9. 
# 9 - AUC Score (CV): 0.7960000529291101  Accuracy (CV): 0.862242720164909

# Recording the AUC and Accuracy of Test Data

#%%
# AUC and Accuracy of Test Data
# test_bow_results = KNN_test(trained_KNN_Model, X_test_bow, Y_test)

#%%
# print(test_bow_results)
# %%
auc_score_tf_idf_test, accuracy_tf_idf_test = KNN_train_simple_cv(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

# print(f"AUC Score (Test): {auc_score_tf_idf_test}")
# print(f"Accuracy (Test): {accuracy_tf_idf_test}")