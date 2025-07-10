# Documentation for `/content/autodocgen/input_code/testing/pipelines/sentiment_pipeline.py`

## FunctionDef: `analyze_sentiment`

 1. Docstring for the `analyze_sentiment` function:

```python
def analyze_sentiment(text: str) -> str:
    """
    Analyzes the sentiment of a given text using a trained sentiment model.
    First, the text is cleaned by removing punctuations and converting to lowercase.
    Then, the text is tokenized using a simple tokenizer.
    Finally, the sentiment of the tokens is predicted using a pre-trained sentiment model.

    Args:
        text (str): The text to analyze for sentiment.

    Returns:
        str: A string representing the sentiment of the input text.

    Examples:
        >>> analyze_sentiment("I love this product! It works great!")
        'Positive'
        >>> analyze_sentiment("This product is terrible. It doesn't work at all.")
        'Negative'
    """
```

2. Short example showing how it's used:

```python
from analyze_sentiment import analyze_sentiment

text = "I love this product! It works great!"
sentiment = analyze_sentiment(text)
print(sentiment)  # Output: Positive
```

This example analyzes the sentiment of a positive text, "I love this product! It works great!", and prints 'Positive' as the sentiment.

