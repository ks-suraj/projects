from models.tokenizer import SimpleTokenizer
from models.sentiment import SentimentModel
from pipelines.text_cleaner import clean_text

def analyze_sentiment(text):
    cleaned = clean_text(text)
    tokens = SimpleTokenizer().tokenize(cleaned)
    sentiment = SentimentModel().predict(tokens)
    return sentiment
