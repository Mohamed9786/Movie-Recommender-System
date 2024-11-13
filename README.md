# Movie-Recommender-System

This project implements a movie recommendation system using Collaborative Filtering with Singular Value Decomposition (SVD). It uses the Surprise library for building the recommendation model, and pandas for data manipulation.

## Project Overview
The goal of this project is to recommend movies to users based on their previous ratings. The system utilizes a dataset containing user ratings for various movies and applies collaborative filtering techniques to make predictions about movies that a user may like.

The model is built using SVD (Singular Value Decomposition), a popular matrix factorization technique used for recommender systems.

## Dependencies
- Pandas: For reading and processing CSV files and handling data.
- Numpy: For numerical operations.
- Surprise: For building and evaluating the recommendation model.

## Dataset
This project uses the following datasets:

* ratings.csv: Contains user ratings for movies.
* movie.csv: Contains movie details, including movie titles.
You can replace these datasets with your own by modifying the file paths in the code.

## How It Works
### 1. Data Loading and Preprocessing:
The ratings and movies datasets are loaded using pandas.
The year of release is extracted from the movie titles and cleaned.
### 2. Model Building:
The data is prepared for training using the Surprise library’s Dataset class.
The training and test sets are split using train_test_split from Surprise.
SVD (Singular Value Decomposition) is used as the collaborative filtering algorithm for making recommendations.
### 3. Recommendations:
The trained model predicts ratings for unrated movies.
A function recommend_movies is defined to recommend the top N movies to a user, based on their previous ratings.
### 4. Evaluation:
The model is evaluated using the Root Mean Square Error (RMSE) metric to assess its accuracy.
