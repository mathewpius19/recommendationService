from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEXING_DIR = BASE_DIR / "indexing"

MODELS_DIR = BASE_DIR / "models"

# Input files (adjust to your dataset)
MOVIES_FILE = RAW_DIR / "movies.csv"
RATINGS_FILE = RAW_DIR / "ratings.csv"
TAGS_FILE = RAW_DIR / "tags.csv"

# Outputs
MOVIE_TEXT_FILE = PROCESSED_DIR / "movie_texts.json"

# TMDB API Key
TMDB_API_KEY = "b196bd7e13603346f3e3cdfc3b7664e2"
