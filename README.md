# Sentiment Prediction from Amazon Reviews
In the vast landscape of e-commerce, understanding customer sentiment is crucial for businesses seeking to enhance user experiences and optimize product offerings. Amazon, being one of the world's largest online marketplaces, accumulates an immense volume of user-generated reviews.This project employs advanced natural language processing and machine learning to analyze Amazon reviews, predicting sentiment shifts. It assists users and businesses in understanding and anticipating public perceptions of products, guiding development, and enhancing decision-making within the e-commerce landscape.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset Description](#dataset-description)
3. [Project Architecture](#project-architecture)
4. [File Structure](#file-structure)
5. [Data Details](#data-details)
6. [Performance Metric](#performance-metric)
7. [Load the Data and Perform Data Analysis](#load-the-data-and-perform-data-analysis)
8. [Top 10 Most Asked Questions on Quora](#top-10-most-asked-questions-on-quora)
9. [Distribution of Question Lengths](#distribution-of-question-lengths)
10. [Feature Engineering](#feature-engineering)
   - [Feature Extraction](#feature-extraction)
     
   - [Processing and Extracting Features](#processing-and-extracting-features)
    
   - [Pre-processing of Text](#pre-processing-of-text)
     
   - [Extracting Features](#extracting-features) 

   - [Visualizing in Lower Dimension using t-SNE](#visualizing-in-lower-dimension-using-t-sne)
   - [Featurizing Text Data with Tf-Idf Weighted Word-Vectors](#featurizing-text-data-with-tf-idf-weighted-word-vectors)
     
11. [Splitting into Train and Test Data](#splitting-into-train-and-test-data)
  
     
12. [Distribution of Output Variable in Train and Test Data](#distribution-of-output-variable-in-train-and-test-data)
  
     
13. [Results](#results)


## Problem Statement:

In the ever-expanding realm of e-commerce, businesses grapple with the challenge of distilling meaningful insights from the vast troves of customer-generated content, particularly in the form of reviews on platforms like Amazon. The problem at hand lies in deciphering the sentiments expressed within these reviews, ranging from glowing endorsements to pointed criticisms. Understanding customer sentiment is pivotal for businesses seeking to enhance product offerings, improve customer experiences, and stay competitive in the dynamic marketplace.The challenge further extends to the sheer volume and diversity of textual data. As the number of reviews grows exponentially, manual analysis becomes impractical. Traditional methods fall short in efficiently extracting sentiments at scale, necessitating the implementation of advanced technologies such as Natural Language Processing (NLP) and Machine Learning (ML).Doing so will empower businesses with actionable insights derived from customer sentiments, enabling them to make informed decisions, improve products and services, and ultimately thrive in the highly competitive landscape of e-commerce.


# About DataSet
This dataset consists of reviews of fine foods from Amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.

- <b>Contents</b>
database.csv : Contains the table 'Reviews'
Reviews.csv : Pulled from the corresponding SQLite table named Reviews in database.sqlite
- <b>Data includes:</b>
Reviews from Oct 1999 - Oct 2012
568,454 reviews
256,059 users
74,258 products
260 users with > 50 reviews

# Performance Metric
Metric(s): 
- AUC Score-Loss
- Binary Confusion Matrix0

# Importing Needed Libraries and accessing other py files(feature-extraction)
The project initiates data analysis and machine learning by importing essential Python libraries, including feature extraction, data visualization, and algorithms. It accesses specific functionalities from the 'feature_extraction' and 'ml_algorithms' modules for further use.


# Load the Data and Perform Data Analysis
Read CSV file into a Pandas DataFrame, display the first five rows and provide information about the dataset. It identifies the column on the basis of scores and dropping value above needed threshold and removing the duplicated values. The dataset initially has 568,454 reviews, and for the fast calculation we used 250,000 and after dropping rows with thresholdlimit its left with 230478 review and after removing the duplicate value 182,285 404,287 reviews. There are many duplicate value and and score above 3, dropping those rows.


# Distribution of data points among output classes (Similar and Non Similar Questions
- <b> Distribution of Duplicate and Non-duplicate reviews </b>

- <b>Number of reviews above score given thresholds</b>
  Analyzing the dataset reveals 250,000 reviews. About 27.08% of questions appear more than score level, with duplication.

- <b>Checking for Duplicates</b>
  No rows are found where 'qid1' and 'qid2' are the same or interchanged, indicating no duplicate question pairs in the dataset.

- <b>cleaning the sentence along with text and words</b>



# Feature Engineering
## Feature Extraction
- **dropping
- **assigning the value
- **removing the duplicate
- **checking using numerator and demonitor 
- **removing the punctuation,HTML tag ,URL and Non-Alpha numeric
- **remove the stop words
- **tokenization of text
- **snowball stemmer for steming of word
- **keeping only aplha-numeric.
- **splitting sentence into words
- **double check for alphanumeric

## Feature Extraction after pre-processing.
Featurization (NLP and Fuzzy Features) Definition:

- <b>Token:</b> You get a token by splitting sentence a space
- <b>Stop_Word:</b> stop words as per NLTK.
- <b>Word:</b> A token that is not a stop_word

