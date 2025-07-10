# 🚀 AutoDocGen — Autonomous Python Documentation Generator (LLM-Powered)

AutoDocGen automatically generates Markdown documentation for your Python projects using **OpenRouter's Llama 4 Maverick (free tier)** model.

This project runs **entirely on Google Colab** — no local setup needed.

---

## ✅ How It Works:
1. Upload your `.zip` Python project (containing `.py` files).
2. Colab unzips the files automatically.
3. The code extracts functions & classes using Python AST.
4. Sends **only small code snippets** (individual functions/classes) to **OpenRouter API**.
5. Generates:
   - Clean docstrings  
   - Code usage examples  
6. Outputs Markdown documentation for each `.py` file.
7. Zips the docs and lets you download them.

---


> 🚀 **Try the full working pipeline in Colab here:**  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15KnqH8i3X9NoedY_4AvaMVR3G2uCWpYp?authuser=0#scrollTo=uMWFehgMHFMo)

---

## ✅ Folder Structure :

  📂 Project Structure — AutoDocGen

  autodocgen/                      // Root project directory
  
    ├── app/                       // Core modules for parsing and documentation
    │   ├── doc_generator.py       // Contains AST parsing and LLM doc generation logic
    │   └── utils.py               // Helper utilities (if extended later)
    
    ├── input_code/                // Folder where uploaded Python code (unzipped) will be stored
    │   └── your_project_files.py  // Your Python source code files
    
    ├── generated_docs/            // Output folder for generated Markdown documentation
    │   └── your_project_files.md  // Auto-generated Markdown documentation per file
    
    ├── autodocgen_pipeline.ipynb  // Main Colab pipeline (end-to-end notebook)
    
    └── README.md                  // Project documentation (this file)


  💡 Explanation:
  - The "app" folder contains your core logic:
    * doc_generator.py → Handles:
        - Parsing Python files using AST
        - Extracting function/class code snippets
        - Sending them to OpenRouter API (Llama 4 Maverick)
        - Generating documentation with docstrings and usage examples
    
    * utils.py → Reserved for helper functions or reusable components if the project scales.

  - "input_code" holds the uploaded project (unzipped automatically inside Colab).

  - "generated_docs" stores the auto-generated documentation as Markdown files.

  - autodocgen_pipeline.ipynb is your main Colab notebook.
    * It handles everything:
        - Upload .zip → Unzip → Parse → Generate docs → Download ZIP.
    * Fully works inside Colab (no local setup needed).
    * Uses OpenRouter API key stored securely via Colab environment variable or secrets.
    * Includes polite rate-limiting to avoid hitting OpenRouter’s free-tier limits.

  - README.md provides project documentation, structure, and usage guide.

  ✅ Clean, modular, Colab-based, with potential for easy extension (e.g., adding Hugging Face Spaces, UIs, or advanced LLMs later).



---

## ✅ **Full Colab Code (Ready to Paste):**

```python
# STEP 1 — Install Dependencies
!pip install requests

# STEP 2 — Set Up OpenRouter API Key (via Colab Secrets or manual)
import os
os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key"  # Replace with your actual key

# STEP 3 — Upload Python Project ZIP
from google.colab import files
print("Upload your .zip Python project:")
uploaded = files.upload()

# STEP 4 — Unzip Uploaded Code
import zipfile

uploaded_zip = list(uploaded.keys())[0]
project_dir = '/content/autodocgen/input_code/'
os.makedirs(project_dir, exist_ok=True)

with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
    zip_ref.extractall(project_dir)

print(f"Codebase extracted to: {project_dir}")

# STEP 5 — Define Parsing & Documentation Functions
import ast
import glob
import requests
import json
import time

def parse_python_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)
    return tree

def extract_code_snippet(filepath, node):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    end_line = getattr(node, 'end_lineno', node.lineno + 10)
    return ''.join(lines[node.lineno - 1:end_line])

def generate_doc_with_openrouter(code_snippet, name):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "X-Title": "AutoDocGen",
    }

    prompt = f"""
You are a senior Python developer. Document the following Python code snippet named '{name}'. Provide:
1. A clear docstring for this function/class.
2. A short example showing how it's used.

Here is the code:
{code_snippet}
"""

    payload = {
        "model": "meta-llama/llama-4-maverick:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        raise Exception(f"OpenRouter API Error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]

# STEP 6 — Process Codebase & Generate Documentation
output_dir = "/content/autodocgen/generated_docs/"
os.makedirs(output_dir, exist_ok=True)

source_files = glob.glob(f"{project_dir}/**/*.py", recursive=True)

for file_path in source_files:
    print(f"Processing: {file_path}")
    tree = parse_python_file(file_path)
    
    markdown_output = f"# Documentation for `{file_path}`\n\n"
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            snippet = extract_code_snippet(file_path, node)
            print(f"Documenting {type(node).__name__}: {node.name}")
            try:
                doc = generate_doc_with_openrouter(snippet, node.name)
                markdown_output += f"## {type(node).__name__}: `{node.name}`\n\n{doc}\n\n"
            except Exception as e:
                print("Error:", e)
            time.sleep(2)  # Prevent hitting rate limits

    output_file = os.path.join(output_dir, os.path.basename(file_path).replace(".py", ".md"))
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(markdown_output)

print("Documentation saved to:", output_dir)

# STEP 7 — Zip and Download Documentation
import shutil
shutil.make_archive("/content/autodocgen/generated_docs", 'zip', output_dir)
files.download("/content/autodocgen/generated_docs.zip")
```

✅ Key Notes:
Runs 100% on Colab + OpenRouter (no local requirements).

Works well for Python 3.8+ (for end_lineno support in AST).

Tested on free-tier OpenRouter API with reasonable delays.

Supports complex multi-file projects (recursive parsing).

✅ License:
MIT License — Free for personal or commercial use.

✅ Author:
Maintained by ks-suraj
