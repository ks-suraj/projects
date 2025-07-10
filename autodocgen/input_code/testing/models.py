class TextClassifier:
    """
    A dummy text classifier for demonstration purposes.
    """

    def __init__(self):
        self.labels = ["positive", "negative", "neutral"]
    
    def predict(self, text):
        """
        Predicts a dummy label based on text length.
        """
        if len(text) > 20:
            return "positive"
        elif len(text) > 10:
            return "neutral"
        else:
            return "negative"
