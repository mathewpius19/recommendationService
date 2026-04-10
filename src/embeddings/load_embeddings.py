import numpy as np
from src.config import EMBEDDINGS_DIR
from sklearn.preprocessing import normalize

def load_embeddings():
    # TODO add checks for file existence and handle exceptions
    embeddingsPath = EMBEDDINGS_DIR / "movie_embeddings.npy"
    idsPath = EMBEDDINGS_DIR / "movie_ids.npy"
    embeddings = np.load(embeddingsPath)
    ids = np.load(idsPath)
    if(embeddings.shape[0] != ids.shape[0]):
        raise ValueError(f"Mismatch in number of rows for datasets at pathts:{idsPath} and {embeddingsPath}")
    if(not np.isfinite(embeddings).all()):
        raise ValueError(f"Embeddings contain non-finite values at path: {embeddingsPath}")
    if(not (isNormalized(embeddings))):
        raise ValueError(f"Embeddings are not normalized:")
    if not passes_self_similarity_test(embeddings):
        raise ValueError(f"Embeddings are not similar:")
    return ids, embeddings

def isNormalized(arr):
    norm = np.linalg.norm(arr, axis=1)
    return np.all(np.isclose(norm,1.0, atol=1e-3))


#O(N × D) N movies with D dimensions, 
# we can compute the cosine similarity of a random embedding with all other embeddings in O(N × D) time. 
# D=384 for all-MiniLM-L6-v2 model, which is manageable for our use case.
#This is a brute force approach of calculating similarities with each vector with itself and checking if the most similar vector is itself and the similarity score is close to 1.0.
def passes_self_similarity_test(embeddings, num_of_tests = 5):
    n = embeddings.shape[0]
    if n==0:
        return False

    for _ in range(num_of_tests):
        i = np.random.randint(0,n)
        
        q = embeddings[i]
        

        #compute cosine similarities
        scores = embeddings @ q
       
        topIdx = np.argsort(scores)[::-1]
        
        #scores cant have values that are infinite or NaN
        if not np.isfinite(scores).all():
            return False
        #post normalization, cosine similarity will always be less than 1 or equal to 1. 
        # If we see values greater than 1, it indicates an issue with the embeddings.
        if scores[i] > 1.01:
            return False
        if topIdx[0]!=i:
            return False
        if not np.isclose(scores[i], 1.0, atol=1e-3):
            return False
    return True
