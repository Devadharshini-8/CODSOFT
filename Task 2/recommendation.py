import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample data: Movie features (Action, Comedy, Drama) and user ratings
movies = np.array([
    [1, 0, 1],  # Movie1: Action, Drama
    [0, 1, 0],  # Movie2: Comedy
    [1, 0, 0],  # Movie3: Action
    [0, 0, 1]   # Movie4: Drama
])
ratings = np.array([4.5, 3.0, 4.0, 3.5])  # Ratings for each movie

# User preferences (e.g., prefers Comedy)
user_prefs = np.array([1, 0, 1])  # [Action, Comedy, Drama]

def recommend_items(user_prefs, movie_features, ratings, min_rating=2.5):
    print(f"User prefers: {user_prefs}")
    similarity_scores = cosine_similarity([user_prefs], movie_features)[0]
    weighted_scores = similarity_scores * ratings
    valid_indices = np.where(ratings >= min_rating)[0]
    if len(valid_indices) > 0:
        valid_similarities = similarity_scores[valid_indices]
        valid_weights = weighted_scores[valid_indices]
        # Find the best match by similarity
        best_match_idx = np.argmax(valid_similarities) if any(valid_similarities > 0) else 0
        best_idx = valid_indices[best_match_idx]
        # Remove best match and sort remaining by rating
        remaining_indices = [i for i in valid_indices if i != best_idx]
        if remaining_indices:
            second_idx = remaining_indices[np.argsort(ratings[remaining_indices])[::-1][0]]
            return [f"Movie{i+1}" for i in [best_idx, second_idx]]
        return [f"Movie{best_idx+1}"]
    return []

# Test the recommendation system
print("Movie features and ratings loaded.")
recommendations = recommend_items(user_prefs, movies, ratings)
print(f"Recommended movies: {recommendations}")
