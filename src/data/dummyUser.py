import pandas as pd
from src.config import DATA_DIR

def load_user_data():
    dummy_data = [
        # movieId, rating, clicks, watched, last_interaction_days, genre_pref
        (1,    5.0, 12, True,  1,  "Animation"),
        (50,   4.5,  8, True,  3,  "Sci-Fi"),
        (318,  4.0,  5, True, 10,  "Drama"),
        (527,  3.5,  2, True, 30,  "War"),
        (593,  2.0,  1, False, 60, "Thriller"),
        (1196, 5.0, 10, True,  2,  "Sci-Fi"),
        (2571, 4.5,  6, True,  7,  "Sci-Fi"),
        (2959, 4.0,  4, True, 14,  "Crime"),
        (4993, 3.0,  3, False, 45, "Fantasy"),
        (7153, 1.0,  1, False,  5,  "Horror"),
    ]

    user_df = pd.DataFrame(dummy_data, columns=[
        "movie_id",
        "rating",
        "clicks",
        "watched",
        "days_since_last_interaction",
        "preferred_genre"
    ])

    # print(user_df)
    return user_df

load_user_data()