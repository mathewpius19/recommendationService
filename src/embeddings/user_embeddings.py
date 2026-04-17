import pandas as pd
import numpy as np
from src.data.dummyUser import load_user_data
from src.data.load import load_data
from src.embeddings.load_embeddings import load_embeddings
from src.util.util import get_indices_for_user,map_moviesids_indexes
import math

user_inteactions = load_user_data()
movieIds,embeddings = load_embeddings()
movie_ids_to_index = map_moviesids_indexes(movieIds)
movies = load_data()

def compute_interaction_weights(df):
    print("computing interaction weights...")
    weights = []
    for row in df.itertuples():
        rating_score =  row.rating/ 5.0
        clicks_score = math.log(1 + row.clicks)
        watched_score = 1 if row.watched else 0.3
        recency_score = math.exp(-(row.days_since_last_interaction) / 30)
        weights.append(rating_score*clicks_score*watched_score*recency_score)
    # print(weights)
    return weights
# compute_interaction_weights(user_inteactions)

def get_user_item_vectors(user_movie_ids):
    indices_for_user = get_indices_for_user(movie_ids_to_index, user_movie_ids)
    if not indices_for_user:
        return np.empty((0, embeddings.shape[1]), dtype=embeddings.dtype)
    return embeddings[np.array(indices_for_user, dtype=np.int64)]



def build_interaction_vector(item_vectors, interaction_weights):
    """
    Build user interaction embedding from item embeddings + weights

    Parameters
    ----------
    item_vectors : np.ndarray
        Shape (k, D) — embeddings of items user interacted with

    interaction_weights : array-like
        Shape (k,) — weight per interaction

    Returns
    -------
    user_vector : np.ndarray
        Shape (D,) — aggregated interaction vector
    """

    print("building interaction vector...")
    # --- Guard: no interactions ---
    if item_vectors is None or item_vectors.size == 0:
        return np.zeros(embeddings.shape[1], dtype=embeddings.dtype)

    weights = np.asarray(interaction_weights, dtype=np.float32)


    # --- Guard: length mismatch ---
    if item_vectors.shape[0] != weights.shape[0]:
        raise ValueError(
            f"Mismatch: {item_vectors.shape[0]} vectors vs {weights.shape[0]} weights"
        )

    # --- Guard: invalid weights ---
    total_weight = weights.sum()
    if total_weight <= 0 or not np.isfinite(total_weight):
        return np.zeros(item_vectors.shape[1], dtype=item_vectors.dtype)

    # --- Weighted aggregation (vectorized) ---
    # Expand weights to (k, 1) for broadcasting
    weighted_vectors = item_vectors * weights[:, np.newaxis]

    # Sum across items → single vector
    user_vector = weighted_vectors.sum(axis=0)

    # Normalize by total weight (true weighted mean)
    user_vector /= total_weight

    return user_vector

item_vectors = get_user_item_vectors(user_inteactions["movie_id"].values)

weights = compute_interaction_weights(user_inteactions)

user_vec = build_interaction_vector(item_vectors, weights)

#cross check logic here
def computer_user_genre_weights(user_df, interaction_weights):
    print("computing user genre weights...")
    print("userdf", user_df)
    print("interaction weights ", interaction_weights)
    genre_strength = {}
    for row, weight in zip(user_df.itertuples(), interaction_weights):
        genres = [g.strip() for g in row.genrePref.split(",")]

        for genre in genres:
            if genre not in genre_strength:
                genre_strength[genre] = 0.0
            genre_strength[genre] += weight
    total = sum(genre_strength.values())

    if total > 0:
        for g in genre_strength:
            genre_strength[g] /= total
    return genre_strength

def build_genre_cetroid(genre, movies_df, max_samples=300):
    print("building genre centroid...")
    mask = movies_df["genres"].str.contains(
        genre,
        case=False,
        na=False
    )

    genre_movie_ids = movies_df.loc[mask, "movie_id"].values

    if len(genre_movie_ids) == 0:
        return None

    if len(genre_movie_ids) > max_samples:
        genre_movie_ids = np.random.choice(
            genre_movie_ids,
            size=max_samples,
            replace=False
        )

    item_vectors = get_user_item_vectors(genre_movie_ids)

    if item_vectors.size == 0:
        return None

    return item_vectors.mean(axis=0)


def build_preference_vector(user_df,
                            user_preferences,
                            movies_df,
                            max_samples=300):

    print("userdf", user_df)
    print("user pref", user_preferences)
    print("movies", movies_df)
    print("inside preference vector...")
    if user_df is None or len(user_df) == 0:
        print("inside first if")
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    if not user_preferences or "genrePref" not in user_preferences:
        print("inside second if")
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    preferred_genres = set(user_preferences["genrePref"])

    # --- Step 1: Compute interaction weights ---
    interaction_weights = compute_interaction_weights(user_df)

    # --- Step 2: Compute user genre weights ---
    genre_weights = computer_user_genre_weights(
        user_df,
        interaction_weights
    )

    print("prefered genres ", preferred_genres)
    print("genre weights ", genre_weights)
    
    # Filter only preferred genres
    genre_weights = {
        g: w for g, w in genre_weights.items()
        if g in preferred_genres
    }
    print("genre weights ", genre_weights)
    if not genre_weights:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    # --- Step 3: Combine weighted centroids ---
    combined = np.zeros(embeddings.shape[1], dtype=np.float32)

    for genre, weight in genre_weights.items():

        centroid = build_genre_cetroid(
            genre,
            movies_df,
            max_samples=max_samples
        )

        if centroid is None:
            continue

        combined += weight * centroid

    # --- Step 4: Normalize ---
    norm = np.linalg.norm(combined)

    if norm > 0 and np.isfinite(norm):
        combined /= norm
    else:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    return combined

def get_filetered_movies(movie_df,user_preferences, maxSamples = 300):
    if not user_preferences or "genres" not in user_preferences:
        print("no genres provided in user preferences")
        return []
    genres = user_preferences["genres"]
    if not genres:
        return []
    pattern = "|".join(genres)
    mask = movie_df["genres"].str.contains(pattern, case=False, na=False)

    watched_movies = movie_df.loc[mask, "movie_id"].values
    if len(watched_movies) > maxSamples:
        watched_movies = np.random.choice(watched_movies, size=maxSamples, replace=False)
    return watched_movies


# preferences = {"genres": ["Sci-Fi"]}

# pref_vec = build_preference_vector(user_inteactions,preferences, movies)

# print("preference vector for user: ", pref_vec)
# scores = embeddings @ pref_vec
# topIndices = np.argsort(scores)[::-1][:5]
# print("top 5 movies for user preferences: ", movies.loc[topIndices, ["title", "genres"]])

def build_user_embedding(user_interaction_vector, preference_vector, alpha=0.7):
    if user_interaction_vector is None or preference_vector is None:
        raise ValueError("Input vectors cannot be None")
    
    if not np.any(user_interaction_vector):
        combined = preference_vector.copy()
    elif not np.any(preference_vector):
        combined = user_interaction_vector.copy()
    else:
        beta = 1 - alpha
        combined = alpha * user_interaction_vector + beta * preference_vector
    norm = np.linalg.norm(combined)

    # Guard against zero or non-finite norm
    if norm > 0 and np.isfinite(norm):
        combined /= norm
    else:
        return None
    
    return combined




