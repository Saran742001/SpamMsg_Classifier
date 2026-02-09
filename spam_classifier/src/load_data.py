import pandas as pd

def load_dataset(path="data/spam.csv"):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )
    return df
