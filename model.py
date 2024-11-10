from surprise import Dataset,Reader,SVD
from surprise.model_selection import cross_validate
import pandas as pd

# Assuming 'ratings' is a DataFrame containing 'userId', 'movieId', and 'rating' columns
# Load the data into Surprise format
ratings = pd.read_csv("ratings_small.csv")

reader=Reader(rating_scale=(0.5,5))
data=Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']],reader)

# Use SVD for matrix factorization
svd = SVD()

# Cross-validation to evaluate the algorithm
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
trainset=data.build_full_trainset()
svd.fit(trainset)


# Predict the rating for a specific user and movie
user_id = 1
movie_id = 10
rating_prediction = svd.predict(user_id, movie_id)
print(f"Predicted rating for user {user_id} and movie {movie_id}: {rating_prediction.est}")


# Function to recommend top N movies for a given user
def recommend_movies(user_id, num_recommendations=10):
    # Get a list of all movie ids
    movie_ids = ratings['movieId'].unique()
    # Predict ratings for all movies the user hasn't rated yet
    movie_ratings = [svd.predict(user_id, movie_id).est for movie_id in movie_ids]
    # Create a DataFrame of movie ids and predicted ratings
    recommendations = pd.DataFrame({
    'movieId': movie_ids,
    'predicted_rating': movie_ratings
    })
    # Sort the DataFrame by predicted rating in descending order
    recommendations = recommendations.sort_values(by='predicted_rating',
    ascending=False)
    # Get the top N recommended movies
    top_recommendations = recommendations.head(num_recommendations)
    # Merge with the movies DataFrame to get movie titles
    top_recommendations = pd.merge(top_recommendations, ratings, on='movieId')
    return top_recommendations
# Recommend top 10 movies for user with ID 1
recommendations = recommend_movies(1, 10)
print(recommendations)
