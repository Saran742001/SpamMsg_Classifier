import re

def clean_text(text: str) -> str:
    """
    Clean text for NLP similarity comparison
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # remove numbers & symbols
    text = re.sub(r"\s+", " ", text)        # remove extra spaces
    return text.strip()
