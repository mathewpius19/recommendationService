from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
from src.config import EMBEDDINGS_DIR
def load_embeddings_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("loaded model successfully... ", model, "\n")
    return model

def encode_texts(model,texts):
    embeddings = model.encode(texts, convert_to_numpy = True,show_progress_bar=True, batch_size=64,)
    print("created embeddings \n")
    return embeddings

def normalize_embeddings(embeddings):
    print("normalizing embeddings... \n")
    normalized = normalize(embeddings, norm='l2')
    print("normalized embeddings \n")
    return normalized

def save_embeddings(ids, embeddings):
    # TODO add checks for ids and embeddings shapes and types
    save_embeddingsPath = EMBEDDINGS_DIR / "movie_embeddings.npy"
    save_IdsPath = EMBEDDINGS_DIR / "movie_ids.npy"
    np.save(save_IdsPath, ids)
    np.save(save_embeddingsPath, embeddings)
    return f"Saved embeddings to {save_embeddingsPath} and ids to {save_IdsPath}"

    
          
