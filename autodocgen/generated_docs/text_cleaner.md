# Documentation for `/content/autodocgen/input_code/testing/pipelines/text_cleaner.py`

## FunctionDef: `clean_text`

 1. Docstring for the `clean_text` function:

```python
def clean_text(text: str) -> str:
    """
    Cleans a given text by removing all non-alphanumeric (non-alphabets and numbers) characters and preserving whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text with only alphanumeric characters and spaces.
    """
```

2. Short example showing how it's used:

```python
text = "Example text! 1234 with  Special Characters - _ , . and  numbers !#@$%^&*()"
cleaned_text = clean_text(text)
print(cleaned_text)  # Output: "Example text1234 with Special Characters and numbers"
```

Note that spaces before and after some non-alphanumeric characters are also removed, so "Example text1234" could also be the output with the given input text example.

