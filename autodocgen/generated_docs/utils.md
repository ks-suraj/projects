# Documentation for `/content/autodocgen/input_code/testing/utils.py`

## FunctionDef: `clean_text`

 1. Docstring for the `clean_text` function:

```python
def clean_text(text: str) -> str:
    """
    This function cleans and normalizes a given text by removing extra spaces, tabs, and numbers, as well as maintaining a single space between words. It returns the cleaned text.

    :param text: string containing the text to be cleaned.
    :return: string containing the cleaned text.
    """
```

2. Short example showing how it's used:

```python
import re

# Here's an example text with extra spaces, numbers, and special characters
example_text = "   This is 1234 a   test 123 with spaces and numbers  12345 and special characters like @#$%^&*()_+-="

# Clean the text using the clean_text function
cleaned_text = clean_text(example_text)
print("Cleaned text:", cleaned_text)

# Output:
# Cleaned text: This is a test with spaces and special characters
```

