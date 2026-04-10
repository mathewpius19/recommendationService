from src.embeddings.load_embeddings import load_embeddings
from src.indexing.initialize_indexes import load_index, index_file
import numpy as np

#load movieIds and embeddings from disk
movie_ids,embeddings = load_embeddings()


#map movieIds to row indexes stored on disk
def map_movies_indexes(movie_ids):
    movie_ids = movie_ids.astype(np.float32)
    indexes = load_index(index_file)
    
    indexesNp = indexes.reconstruct_n(0, len(movie_ids)).astype(np.float32)
    if(len(movie_ids)!=len(indexesNp)):
        return f"movies and indexes dont match in length {len(movie_ids)}, {len(indexes)}"
    else:
        return {movie_id: index for movie_id, index in zip(movie_ids, indexesNp)}
    
def map_moviesids_indexes(movie_ids):
    return {movie_id:idx for idx,movie_id in enumerate(movie_ids)}
            
#get movie Ids from user to map corresponding indices stored on disk
def get_indices_for_user(movie_ids_to_index, user_movie_ids):
    indicies = []
    for movie_id in user_movie_ids:
        if movie_id in movie_ids_to_index:
            indicies.append(movie_ids_to_index[movie_id])
    return indicies
