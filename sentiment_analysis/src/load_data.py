import pandas as pd

def load_dataset(path="data/imdb_reviews.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df

