def read_text_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def write_result(filepath, result):
    with open(filepath, 'w') as f:
        f.write(result)
