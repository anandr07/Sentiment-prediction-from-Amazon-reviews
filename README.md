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





# Sentiment Prediction from Amazon Reviews

This project employs advanced natural language processing and machine learning to analyze Amazon reviews, predicting sentiment shifts. It assists users and businesses in understanding and anticipating public perceptions of products, guiding development, and enhancing decision-making within the e-commerce landscape.

## Problem Description:

### Contents

- [Overview](#overview)
- [Data](#data)
  - [Files](#files)
  - [Details](#details)

## Overview




Brief description of your project.

## Data

### Files

- **database.csv**: Contains the table 'Reviews'
- **Reviews.csv**: Pulled from the corresponding SQLite table named Reviews in database.sqlite

### Details

The dataset includes:
- Reviews from Oct 1999 - Oct 2012
- 568,454 reviews
- 256,059 users
- 74,258 products
- 260 users with > 50 reviews

## Usage

Instructions on how to use your project or any additional information.

## Acknowledgments

- Any credits or acknowledgments you want to include.

