import numpy as np
from src.data.load import load_data
from src.data.preprocess import buildTextColumn
from src.embeddings.embedding_initialization import load_embeddings_model, encode_texts, normalize_embeddings, save_embeddings 
def build_movie_embeddings():
    df = load_data()
    df = buildTextColumn(df)

    model = load_embeddings_model()
    embeddings = encode_texts(model,df["text"])
    embeddings = normalize_embeddings(embeddings)
    # print(embeddings.shape, df["movieId"].to_numpy().shape ,np.isnan(embeddings).any())
    save_embeddings(df["movie_id"], embeddings)

# if __name__ == "__main__":
#     build_movie_embeddings()