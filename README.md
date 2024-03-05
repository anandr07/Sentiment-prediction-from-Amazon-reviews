# Sentiment Prediction from Amazon Reviews
In the vast landscape of e-commerce, understanding customer sentiment is crucial for businesses seeking to enhance user experiences and optimize product offerings. Amazon, being one of the world's largest online marketplaces, accumulates an immense volume of user-generated reviews.This project employs advanced natural language processing and machine learning to analyze Amazon reviews, predicting sentiment shifts. It assists users and businesses in understanding and anticipating public perceptions of products, guiding development, and enhancing decision-making within the e-commerce landscape.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [About DataSet](#About-DataSet)
3. [Performance Metric](#Performance-Metric)
4. [File Structure](#file-structure)
5. [Performance Metric](#performance-metric)
6. [Load the Data and Perform Data Analysis](#load-the-data-and-perform-data-analysis)
7. [Distribution of data points among output classes](#Distribution-of-data-points-among-output-classes)
8. [Feature Engineering](#feature-engineering)
   - [Data Preprocessor](#Data-preprocessor)
   - [Feature Extraction after pre-processing](#Feature-Extraction-after-pre-processing)
   - [Some Additional Feature](#Some-additional-feature)
9. [Splitting into Train and Test Data](#splitting-into-train-and-test-data)    
10. [Models used:](#Models-used)     
11. [Results](#results)


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
- Log-Loss
- Binary Confusion Matrix0

# Importing Needed Libraries and accessing other py files(feature-extraction)
The project initiates data analysis and machine learning by importing essential Python libraries, including feature extraction, data visualization, and algorithms. It accesses specific functionalities from the 'feature_extraction' and 'ml_algorithms' modules for further use.


# Load the Data and Perform Data Analysis
Read CSV file into a Pandas DataFrame, display the first five rows and provide information about the dataset. It identifies the column on the basis of scores and dropping value above needed threshold and removing the duplicated values. The dataset initially has 568,454 reviews, and for the fast calculation we used 250,000 and after dropping rows with thresholdlimit its left with 230478 review and after removing the duplicate value 182,285 404,287 reviews. There are many duplicate value and and score above 3, dropping those rows.


# Distribution of data points among output classes
- <b> Distribution of Duplicate and Non-duplicate reviews </b>

- <b>Number of reviews above score given thresholds</b>
  Analyzing the dataset reveals 250000 unique reviews. About 27.08% of reviews appear more than once.
  
- <b>Checking for Duplicates</b>

- <b>cleaning the sentence along with text and words</b>


# Feature Engineering
## Data preprocessor
- **dropping**
- **assigning the value**
- **removing the duplicate**
- **checking using numerator and demonitor**
- **removing the punctuation,HTML tag ,URL and Non-Alpha numeric**
- **remove the stop words**
- **tokenization of text**
- **snowball stemmer for steming of word**
- **keeping only aplha-numeric.**
- **splitting sentence into words**
- **double check for alphanumeric**

## Feature Extraction after pre-processing.
Featurization (NLP and Fuzzy Features) Definition:

- <b>Token:</b> You get a token by splitting sentence a space
- <b>Stop_Word:</b> stop words as per NLTK.
- <b>Word:</b> A token that is not a stop_word


## Some additional features
- **Bag of words**
- **Uni,Bi and tri grams**
- **Tf-Idf Vectorization**
- **diff_chars**
- **word2vec Model**
- **Average word2vec**
- **Tf-Idf Word2vec**


### Due to lack of Computation Power the models are trained on 250,000 Rows.

- Checks if there are any NA (missing) values in the DataFrame after converting features to numeric format. If present, it prints "NA Values Present"; otherwise, it prints "No NA Values Present." It then displays the number of NaN values in each column after the conversion. Additionally, it converts the target variable y_true to a list of integers and shows the first few rows of the DataFrame.

### Comparing the Original Text and the processed text
![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/73d5b562-2d27-4715-b11d-5929430c3de4)


### Get the top 10 words most similar words to "quality"
![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/de07f898-c08e-4bec-8d2d-6e72c2db0326)


## Splitting into Train and Test Data
Train Data : 70%
Test Data : 30%

## Models used:

### KNN on Bag of words

![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/312bc6ea-d428-4abc-b592-230a46e8a507)
![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/ddecfcd3-bd37-43af-b7a6-19e2868fcca8)
![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/56573d16-cb01-48bf-ad57-84931be29b82)

### KNN on tfidf

![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/a51f3675-0c60-4e88-9bc7-c47427723c7d)
![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/d4b37b3c-0f8f-4a59-bd96-824474cdcd41)
![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/bbbfd75a-10d2-4839-89a5-1874a46191db)

### Naive Bayes on Bag of Words

![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/668cb884-5c3e-48b9-8d91-02a34a03b017)

### Naive Bayes on Tf-Idf

![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/c71d1599-c453-4910-ae9f-7bfec04a77ed)

### SGD Classifier on Bag of Words

![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/eb14b3e1-9851-4139-908d-81426c837119)

### SGD Classifier on Tf-Idf 

![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/6c48f524-3a73-4b72-9c00-9cd991968af5)

### SGD Classifier on word2vec

![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/6b1b2a8d-52bf-4c5a-ac81-b11a6642a8c5)

## Results

- The SGD Classifier on tf-idf give best accuracy among all the model used .
- Best Hyperparameters: {'alpha': 1.9211659757411964e-06, 'eta0': 0.01}
- AUC Score (CV): 0.9582384628250535  Accuracy (CV): 0.9266463021839869
- AUC Score (Train): 0.9995024371625396  Accuracy (Train): 0.9932857742341586
  
 ![image](https://github.com/anandr07/Sentiment-prediction-from-Amazon-reviews/assets/66896800/11aef25f-de02-413e-8014-b0a85da71aab)












