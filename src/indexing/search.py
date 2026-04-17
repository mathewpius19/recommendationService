from src.embeddings.embedding_initialization import load_embeddings_model, encode_texts, normalize_embeddings
from src.service.app_context import app_context
from src.embeddings.user_embeddings import compute_interaction_weights, get_user_item_vectors, build_interaction_vector, build_preference_vector, build_user_embedding
import numpy as np
from src.data.dummyUser import load_user_data

user_inteactions = load_user_data()


def search_query(contentTypes, query):
    model = app_context.model
    movie_ids = app_context.movie_ids
    movies_by_ids = app_context.movies_by_id

    embeddings = encode_texts(model,query)
    normalized_query = normalize_embeddings(embeddings)
    indexes = app_context.index
    distances,indices = indexes.search(normalized_query, k=50)
    row_indices = indices[0]
    # print("row indices: ", row_indices)
    # row_distances = distances[0]    
    movies_with_set_keys = app_context.movies_by_id # do this for faster lookup
    print(f"top 5 results for query \n", query, "are: \n")
    results = []
    movies = []
    documentaries = []
    for rowIndex in row_indices:

        movie_id = movie_ids[rowIndex]
        movie = movies_with_set_keys.loc[movie_id]
        genres = movie["genres"]
        
        if contentTypes:
            if "movies" in contentTypes and "Documentary" not in genres:
                movies.append(int(movie_id))
            if "Documentary" in contentTypes and "Documentary" in genres:
                documentaries.append(int(movie_id))
        else:
            movies.append(int(movie_id))
            
    if "movies" in contentTypes and "Documentary" not in contentTypes:
        for id in movies[0:10]:
            
            results.append(id)

    elif "Documentary" in contentTypes and "movies" not in contentTypes:
        for id in documentaries[0:10]:
            
            results.append(id)
    elif "movies" in contentTypes and "Documentary" in contentTypes:
        for id in movies[0:5]:
            results.append(id)
        for id in documentaries[0:5]:
        
            results.append(id)
    else:
        for id in movies[0:10]:

            results.append(id)

            
    return {"movieIds":results}

def search_user_query(user_df,
                       k=10,
                       alpha=0.7):
    if user_df is None or len(user_df) == 0:
        return []

    genre_pref = user_df["genrePref"].iloc[0] if "genrePref" in user_df.columns else ""
    preferences = {
        "genrePref": [g.strip() for g in genre_pref.split(",") if g.strip()]
    }
    movies_df = app_context.movies 
    index = app_context.index
    movie_ids =  app_context.movie_ids

    interaction_df = user_df[user_df["movie_id"].notna()].copy()
    seen_movie_ids = interaction_df["movie_id"].values if len(interaction_df) > 0 else np.array([])

    if len(seen_movie_ids) == 0:
        preferred_genres = preferences["genrePref"]
        if not preferred_genres:
            return []

        pattern = "|".join(preferred_genres)
        genre_matches = movies_df["genres"].str.contains(pattern, case=False, na=False)
        fallback_ids = movies_df.loc[genre_matches, "movie_id"].dropna().astype(int).tolist()
        return fallback_ids[:k]

    item_vectors = get_user_item_vectors(seen_movie_ids)
    weights = compute_interaction_weights(interaction_df) if len(interaction_df) > 0 else []
    print("computer weights", weights)

    interaction_vec = build_interaction_vector(item_vectors, weights)

    pref_vec = build_preference_vector(user_df,preferences, movies_df)

    user_vec = build_user_embedding(
        interaction_vec,
        pref_vec,
        alpha=alpha
    )
    if user_vec is None:
        return []
    
    # --- FAISS search ---
    query = user_vec.astype(np.float32).reshape(1, -1)
    fetch_k = max(k + len(seen_movie_ids), k)

    distances, indices = index.search(query, fetch_k)

    indices = indices[0]

    recommendations = []

    for idx in indices:
        movie_id = movie_ids[idx]

        if movie_id not in seen_movie_ids:
            # recommendation = app_context.movies_by_id.loc[movie_id][["title","genres"]], "\n"
            recommendations.append(movie_id)

        if len(recommendations) == k:
            break
            
    return recommendations


# def self_query_test():
#     index = np.random.randint(0,embeddings.shape[0])
#     print("index = ",index)

#     score = embeddings @ embeddings[index]
#     topIdx = np.argsort(score)[::-1]
#     print(score[topIdx])
#     return "top 5", topIdx[:5]

# queries = np.array([["movies that are action and comedy"], ["animated movie about toys"], ["documentaries about dinosaurs"]])
# for query in queries:
#     search_query(query)

# print(search_user_query(user_inteactions, {"genres": ["Sci-Fi", "Drama"]}, app_context.movies, app_context.index, app_context.movie_ids))
