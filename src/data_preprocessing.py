#%%[markdown]
# This file contains the text-processing functions needed for cleaning and processing of text.

#%%
# Importing the needed libraries

import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

#%%
def clean_text(text):
    ''' This function removes punctutations, HTML tags, URLs, and Non-Alpha Numeric words.
    '''
    unwanted_chars_patterns = [
        r'[!?,;:—".]',  # Remove punctuation
        r'<[^>]+>',  # Remove HTML tags
        r'http[s]?://\S+',  # Remove URLs
        r"^[A-Za-z]+$" # Non-Alpha Numeric
    ]
    
    for pattern in unwanted_chars_patterns:
        text = re.sub(pattern, '', text)
    
    return text

#%%

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
nltk.download('punkt')

def preprocess_text(text):
    ''' This function performs tokenization of text and also uses Snowball Stemmer for stemming of words.
    '''
    # Tokenizing the text and removing stopwords
    tokens = nltk.word_tokenize(text)
    # tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in stop_words and word.isalpha() and len(word) >= 3]
    # Applying Snowball stemming
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)


#%%
def sentence_to_words(data_frame, column_name):
    ''' This function converts a sentance in words keeping words that are alpha-numeric only.
        Also makes all the words to lowercase
    '''
    i = 0
    list_of_words_in_sentance = []

    for sent in data_frame[column_name].values:
        list_of_words_in_filtered_sentence = []
        sent = clean_text(sent)
        
        # Split the sentence into words
        words = sent.split()

        # Check if each word is alphanumeric (Just for a double check)
        for word in words:
            if word.isalnum():
                list_of_words_in_filtered_sentence.append(word.lower())
        
        list_of_words_in_sentance.append(list_of_words_in_filtered_sentence)

    return list_of_words_in_sentance

    