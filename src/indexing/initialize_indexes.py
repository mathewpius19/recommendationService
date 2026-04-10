from src.config import INDEXING_DIR
from src.embeddings.load_embeddings import load_embeddings
from src.embeddings.embedding_initialization import load_embeddings_model,encode_texts,normalize_embeddings
import numpy as np
import faiss

movieIds,embeddings = load_embeddings()
index_file = INDEXING_DIR / "movie_indices.faiss"

def build_flat_indexes():
    dimension = embeddings.shape[1]
    vectors = embeddings.astype(np.float32)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    save_index(index,index_file)
    return index

def save_index(index, file):
    faiss.write_index(index, str(file))
    
    

def load_index(file):
    return faiss.read_index(str(file))

