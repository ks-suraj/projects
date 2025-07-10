class SentimentModel:
    def predict(self, tokens):
        score = sum(len(token) for token in tokens)
        return "Positive" if score % 2 == 0 else "Negative"
