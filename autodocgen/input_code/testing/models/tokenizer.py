class SimpleTokenizer:
    def __init__(self, lower=True):
        self.lower = lower

    def tokenize(self, text):
        if self.lower:
            text = text.lower()
        return text.split()
