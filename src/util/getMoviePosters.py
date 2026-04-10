import pandas as pd
import requests
import time
from src.config import TMDB_API_KEY,PROCESSED_DIR
API_KEY = TMDB_API_KEY

TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

INPUT_FILE = PROCESSED_DIR / "cleanedMovies.csv"
OUTPUT_FILE =  PROCESSED_DIR / "cleanedMovies.csv"
session = requests.Session()

def fetch_poster(tmdb_id, retries=5):

    if tmdb_id == -1:
        return None

    url = TMDB_MOVIE_URL.format(tmdb_id)
    params = {"api_key": API_KEY}

    for attempt in range(retries):

        try:
            r = session.get(url, params=params, timeout=5)

            if r.status_code == 200:

                data = r.json()
                poster_path = data.get("poster_path")

                if poster_path:
                    return POSTER_BASE_URL + poster_path

                return None

            # retry on temporary TMDB errors
            if r.status_code in [429, 500, 502, 503]:
                time.sleep(2 ** attempt)
                continue

            return None

        except requests.exceptions.RequestException:

            if attempt == retries - 1:
                return None

            time.sleep(2 ** attempt)

    return None


def enrich_posters():

    movies = pd.read_csv(INPUT_FILE)

    poster_urls = []

    for i, row in movies.iterrows():
        tmdb_id = row["tmdbId"]

        poster = fetch_poster(tmdb_id)

        poster_urls.append(poster)

        # avoid rate limits
        time.sleep(0.25)

        if i % 100 == 0:
            print(f"processed {i}/{len(movies)}")

    movies["poster_url"] = poster_urls

    movies.to_csv(OUTPUT_FILE, index=False)

    print("Poster enrichment complete")

def missing_movies():
    movies = pd.read_csv(INPUT_FILE)
    missingMovies = movies[
        movies["poster_url"].isna() | (movies["poster_url"]=="")
        ]
    # print("missing movies count ", len(missingMovies))

    for i,row in missingMovies.iterrows():
        print("fetching for movieId ", row["movie_id"])
        tmdb_id = row["tmdbId"]
        posterUrl = fetch_poster(tmdb_id)
        if posterUrl:
            movies.at[i, "poster_url"] = posterUrl
        time.sleep(0.1)

        if i % 100 == 0:
            print(f"processed {i}")
        

        if i % 500 == 0:
            print("writing into csv....")
            movies.to_csv(INPUT_FILE, index=False)

    movies.to_csv(INPUT_FILE, index=False)
    print("poster enrichment complete")

if __name__ == "__main__":
    # enrich_posters()
    missing_movies()

