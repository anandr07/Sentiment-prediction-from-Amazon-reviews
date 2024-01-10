#%%[markdown]
## Sentiment prediction from Amazon reviews

### About DataSet

# This dataset consists of reviews of fine foods from Amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.

### Contents

# - **database.csv :** Contains the table 'Reviews'
# - **Reviews.csv :** Pulled from the corresponding SQLite table named Reviews in database.sqlite

### Data includes:

# - Reviews from Oct 1999 - Oct 2012 
# - 568,454 reviews 
# - 256,059 users 
# - 74,258 products 
# - 260 users with > 50 reviews

#%%
import sys
import os

# Getting the current script's directory
current_dir = os.getcwd()

# Adding the parent directory to the Python path
sys.path.append(os.path.dirname(current_dir))


#%%
#Importing Libraries

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
from ml_algorithms.NaiveBayes import NaiveBayes_train_simple_cv
from ml_algorithms.SGDClassifier import SGDClassifier_train_random_search_cv
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer


#%%
raw_data = pd.read_csv("E:\Data Science Projects\Sentiment-prediction-from-Amazon-reviews\data\Reviews.csv")

#%%
raw_data.head(10)

#%%
print(raw_data["Text"].head(10))

#%%
# Just for faster computation use first 50000 rows 
# ************************************Remove Later********************************************
raw_data = raw_data[:10000]
# ********************************************************************************************

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

# %%
from wordcloud import WordCloud

# Separate positive and negative reviews
positive_reviews = raw_data[raw_data['Review'] == 'Positive']
negative_reviews = raw_data[raw_data['Review'] == 'Negative']

# Join the cleaned text for positive and negative reviews
positive_text = ' '.join(positive_reviews['Clean_Text'])
negative_text = ' '.join(negative_reviews['Clean_Text'])

# Create WordCloud objects for positive and negative reviews
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
negative_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

# Plot word clouds for positive and negative reviews
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Reviews Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Reviews Word Cloud')
plt.axis('off')

plt.tight_layout()
plt.show()

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

# %%

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

print("Shape of train dataset after converting into tf-idf is ", X_train_tf_idf_vectorizer.get_shape())
print("Shape of test dataset after converting into tf-idf is ", X_test_tf_idf_vectorizer.get_shape())

#%%
# Normalize Tf-Idf Train and Test Data
X_train_tfidf=preprocessing.normalize(X_train_tf_idf_vectorizer)
X_test_tfidf=preprocessing.normalize(X_test_tf_idf_vectorizer)
print("Train Data Size: ",X_train_tfidf.get_shape())
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
## Tf-Idf Word2vec 

#%%
tfidf_model = TfidfVectorizer()
tf_idf_matrix = tfidf_model.fit_transform(raw_data['Clean_Text'].values)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names_out(), list(tfidf_model.idf_)))

# TF-IDF weighted Word2Vec
tfidf_feat = tfidf_model.get_feature_names_out() # tfidf words/col-names
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

#%%[markdown]
# KNN on Bag of Words

#%%
auc_score_bow_test_KNN, accuracy_bow_test_KNN = KNN_train_simple_cv(X_train_bow, Y_train, X_test_bow, Y_test)

#%%[markdown]
# KNN on tf-idf

#%%
auc_score_tf_idf_test_KNN, accuracy_tf_idf_test_KNN = KNN_train_simple_cv(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

#%%[markdown]
# KNN on Tf-Idf word2vec 

#%%
auc_score_word2vec_test_KNN, accuracy_bword2vec_test_KNN = KNN_train_simple_cv(X_train_tfidf_word2vec, Y_train_tfidf_wor2vec, X_test_tfidf_word2vec, Y_test_tfidf_wor2vec)

#%%
# Naive Bayes from here

#%%[markdown]
# NaiveBayes on Bag of Words

#%%
auc_score_bow_test_NB, accuracy_bow_test_NB = NaiveBayes_train_simple_cv(X_train_bow, Y_train, X_test_bow, Y_test)

#%%[markdown]
# NaiveBayes on tf-idf

# %%
auc_score_tf_idf_test_NB, accuracy_tf_idf_test_NB = NaiveBayes_train_simple_cv(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

#%%[markdown]
# NaiveBayes on Tf-Idf word2vec 

#%%
# auc_score_tfidf_word2vec_test_NB, accuracy_tfidf_word2vec_test_NB = NaiveBayes_train_simple_cv(X_train_tfidf_word2vec, Y_train_tfidf_wor2vec, X_test_tfidf_word2vec, Y_test_tfidf_wor2vec)

#%%
# SGD Classifier from here

#%%[markdown]
# SGD Classifier on Bag of Words

#%%
auc_score_bow_test_SGDC, accuracy_bow_test_SGDC = SGDClassifier_train_random_search_cv(X_train_bow, Y_train, X_test_bow, Y_test)

#%%[markdown]
# SGD Classifier on tf-idf

# %%
auc_score_tf_idf_test_SGDC, accuracy_tf_idf_test_SGDC = SGDClassifier_train_random_search_cv(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

#%%[markdown]
# SGD Classifier on Tf-Idf word2vec 

#%%
auc_score_tfidf_word2vec_test_SGDC, accuracy_tfidf_word2vec_test_SGDC = SGDClassifier_train_random_search_cv(X_train_tfidf_word2vec, Y_train_tfidf_wor2vec, X_test_tfidf_word2vec, Y_test_tfidf_wor2vec)

#%%[markdown]

## KNN, Naive Bayes, and SGD Completed 

# %%
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Sample data (replace this with your actual X_train, Y_train data)
# Assuming X_train contains text data and Y_train contains labels (1 for positive, 0 for negative)
# Make sure to replace this with your actual data
raw_data = pd.DataFrame({'Text': X_train[:50000], 'Review': Y_train[:50000]})

# Drop rows with missing or NaN values in the 'Text' column
raw_data = raw_data.dropna(subset=['Text'])

# Function to generate and display word cloud
def generate_word_cloud(text, title):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    
    # Specify the path to a font file available on your system
    font_path = "C:/Windows/Fonts/Arial.ttf"  # Replace with the path to a font file on your system
    
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10,
                          font_path=font_path)  # Specify the font path
    
    # Generate the word cloud
    wordcloud.generate(text)

    # Plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.title(title)
    plt.show()

# Function to get the top N most used words
def get_top_words(text, n=10):
    words = re.findall(r'\b\w+\b', text)
    word_freq = nltk.FreqDist(words)
    return word_freq.most_common(n)

# Concatenate positive and negative reviews using apply and str.cat
positive_reviews_text = raw_data[raw_data['Review'] == 1]['Text'].apply(lambda x: str(x)).str.cat(sep=' ')
negative_reviews_text = raw_data[raw_data['Review'] == 0]['Text'].apply(lambda x: str(x)).str.cat(sep=' ')

# Generate word clouds and display top 10 words
generate_word_cloud(positive_reviews_text, 'Word Cloud for Positive Reviews')
generate_word_cloud(negative_reviews_text, 'Word Cloud for Negative Reviews')

top_positive_words = get_top_words(positive_reviews_text)
top_negative_words = get_top_words(negative_reviews_text)

print("Top 10 Most Used Positive Words:")
print(top_positive_words)

print("\nTop 10 Most Used Negative Words:")
print(top_negative_words)

# %%
