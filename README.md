# 🔍 AutoRAG-Agent

AutoRAG-Agent is a **modular Retrieval-Augmented Generation (RAG) pipeline** designed for document ingestion, semantic search, and answer generation using Large Language Models (LLMs).

It supports:
- ✅ PDF & Web Document Ingestion
- ✅ Text Cleaning & Chunking
- ✅ Embedding + Vector Indexing (FAISS)
- ✅ Semantic Search
- ✅ Answer Generation via LLMs  
With **GPU acceleration** (works in **Google Colab Free Tier** too).

✅ Built for ease of use in **Google Colab**.

---

## 🚀 Try it on Colab

You can run and test this project directly on Google Colab by clicking below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HHCyOO93v13nA8JTJagmiDW1kVIvkErX#scrollTo=EbZq5wYRNx8i)

---
---

## 📂 Directory Structure
/*
Project Directory Structure: AutoRAG-Agent

/*
  📂 Project Structure — AutoRAG-Agent
  
  AutoRAG-Agent/               // Root project directory
  
    ├── app/                  // Core modules for RAG pipeline
    │   ├── ingestion.py          // Document ingestion: PDFs, URLs, and raw text
    │   ├── preprocessing.py      // Text cleaning and chunking (LangChain splitter)
    │   ├── embedding.py          // Embedding text chunks + FAISS vector indexing
    │   ├── query.py              // Semantic search over FAISS index
    │   └── answering.py          // Answer generation via LLM (Flan-T5 or similar)
    │
    ├── sample_data/          // Sample PDF documents for testing
    │   └── sample_doc.pdf        // Example PDF downloaded from arXiv
    │
    ├── models/               // (Optional) Fine-tuned or downloaded models (future use)
    │
    ├── ui/                   // (Optional) UI-related code like Gradio, Streamlit, etc.
    │
    └── README.md             // Project documentation (this file)

  💡 Explanation:
  - `app/` contains all the core modules of the RAG pipeline:
    * ingestion.py → Loads and extracts text from PDFs, URLs, or raw text.
    * preprocessing.py → Cleans and chunks text for embeddings.
    * embedding.py → Embeds text using Sentence Transformers and stores vectors in FAISS.
    * query.py → Performs semantic similarity search over FAISS index.
    * answering.py → Uses a lightweight LLM to answer user queries from retrieved chunks.

  - `sample_data/` contains sample PDFs (like arXiv papers) for testing the pipeline.

  - `models/` can be used to store fine-tuned models or downloaded checkpoints later.

  - `ui/` is reserved for optional future user interfaces (e.g., Gradio or Streamlit apps).

  - The pipeline is modular and ready to be run in notebooks or scripts.
  
  ✅ Clean, GPU-accelerated, modular pipeline ready for research, testing, and integration.


---

## 🚀 Features & Pipeline Workflow

| Step              | Description |
|-------------------|-------------|
| Ingestion         | Load text from PDFs, URLs, or raw text |
| Preprocessing     | Clean and split text into chunks |
| Embedding & Indexing | Embed text chunks with Sentence Transformers and store in FAISS index |
| Semantic Search   | Search FAISS index for relevant chunks |
| Answer Generation | Generate answers using lightweight LLMs (Flan-T5) |

---

## 📋 Setup & Requirements (Google Colab Compatible)


!pip install pypdf requests beautifulsoup4 langchain sentence-transformers faiss-gpu transformers
Go to Runtime → Change runtime type → Select GPU (T4/A100 recommended).

This setup uses faiss-gpu for fast similarity search on GPU.


📝 Module Details & Functionality
1. app/ingestion.py
Ingest text from:

PDF files (pypdf)

URLs (requests + BeautifulSoup)

Raw text strings

2. app/preprocessing.py
Clean text (remove line breaks and tabs).

Chunk text using LangChain's RecursiveCharacterTextSplitter.

3. app/embedding.py
Embed text chunks using sentence-transformers/all-MiniLM-L6-v2.

Store embeddings in FAISS index for fast retrieval.

4. app/query.py
Perform semantic search on FAISS index to retrieve relevant chunks.

5. app/answering.py
Generate answers using lightweight LLMs like google/flan-t5-small.

⚠️ Notes
Adjust chunk size & overlap in preprocessing.py for large documents.

Easily swap out LLMs and Embedding Models (just change model names).

This pipeline is ready for integration with Gradio/FastAPI for interactive UI.

📈 Future Improvements
FAISS index saving/loading from disk

Multi-document ingestion support

Web-based or Command-line UI

Support for advanced LLMs like Mistral, Llama 3, etc.

📚 References
LangChain

FAISS

Sentence Transformers

Transformers (Hugging Face)

🤝 License
MIT License

✅ Conclusion
AutoRAG-Agent is a simple, modular, and GPU-accelerated RAG pipeline for document ingestion, semantic search, and LLM-based answering, ready for both research and practical use cases.
