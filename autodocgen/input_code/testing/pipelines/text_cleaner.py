def clean_text(text):
    return ''.join(c for c in text if c.isalnum() or c.isspace())
