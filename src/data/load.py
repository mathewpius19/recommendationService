import pandas as pd
from src.config import PROCESSED_DIR

DATA_PATH = PROCESSED_DIR / "cleanedMovies.csv"
print("data path is:",DATA_PATH)
def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully. Data info: ",
              df.info())

    return df

