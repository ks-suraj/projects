import re

def clean_text(text):
    """
    Cleans text by removing extra spaces and numbers.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d+', '', text)
    return text
