# 🧠 KS-Suraj | Advanced AI, RAG & Infra Projects

This repository is a portfolio of autonomous systems, AI research tools, and cloud optimization engines — each isolated in a separate Git branch.


---

## 📁 Project Index

| Project | Description | Branch | Technologies Used |
|--------|-------------|--------|-------------------|
| **🌩️ AETHER-AI** | Multi-cloud anomaly detection, RCA, and cost optimization powered by LLMs, LangGraph, and Reinforcement Learning. Outputs visual insights via Streamlit and FastAPI. | [`AETHER-AI`](https://github.com/ks-suraj/projects/tree/AETHER-AI) | LangGraph, LangChain, LlamaIndex, SentenceTransformers, OpenRouter, FAISS, Pinecone, Chroma, Weaviate, RLlib, Stable-Baselines3, MLflow, W&B, Streamlit, FastAPI, Docker, Kubernetes YAML, Google Colab, Git, GitHub |
| **📝 AutoDocGen** | Automatically generates clean Markdown documentation for any Python codebase using AST + LLM (Mistral-7b via OpenRouter). All inside Colab. | [`AUTODOCGEN`](https://github.com/ks-suraj/projects/tree/AutoGenDoc) | AST, SentenceTransformers, OpenRouter, Markdown, Python zipfile/glob, Google Colab, Git, GitHub |
| **🔍 AutoRAG-Agent** | Modular Retrieval-Augmented Generation pipeline for PDFs and web content. Includes ingestion, vector indexing, semantic search, and QA using lightweight LLMs. | [`AUTORAG-AGENT`](https://github.com/ks-suraj/projects/tree/AutoRAG-Agent) | FAISS, LangChain, SentenceTransformers, Flan-T5, PyPDF2, BeautifulSoup, Transformers, Google Colab, Git, GitHub |
| **🧬 Genesis** | Experimental AI system that simulates an autonomous ML research engineer. Parses research, drafts model code, and runs refinement loops offline. | [`GENESIS`](https://github.com/ks-suraj/projects/tree/Genesis-autonomous-ai-engineer) | Python, Modular Agents, Code Generation Logic, Git, GitHub |
| **🔥 RAG-Agent-Ops** | Lightweight RAG pipeline focused on Pinecone integration with simple PDF upload and semantic search via Colab widgets. Ideal for demo or production prototyping. | [`RAG-AGENT-OPS`](https://github.com/ks-suraj/projects/tree/RAG-Agent-Ops) | Pinecone, SentenceTransformers, PyPDF2, Colab Widgets, Google Colab, Git, GitHub |

---

## 🧠 About Each Project

---

### 🌩️ AETHER-AI  
A fully autonomous pipeline for **multi-cloud infrastructure monitoring**, **root cause analysis**, and **cost optimization**.

Includes:
- LLM-based anomaly detection and RCA (LangChain + OpenRouter)
- Reinforcement learning agent for cost-saving actions
- Colab-ready end-to-end pipeline with Streamlit + FastAPI interface
- Multi-vector store compatibility: Pinecone, FAISS, Chroma, Weaviate
- Modular agent design (Monitoring, RCA, Optimizer, Orchestrator)

> Built for enterprise cloud ops teams needing resilience, explainability, and automation.

---

### 📝 AutoDocGen  
An autonomous Python documentation generator that runs 100% in **Google Colab**.

Includes:
- AST parsing of uploaded `.py` files
- Docstring and usage generation via LLM (Llama 4 Maverick / OpenRouter)
- Markdown doc creation per file + ZIP download
- Secure API handling + polite rate-limiting

> Ideal for developers who want fast, explainable docs for personal or professional projects.

---

### 🔍 AutoRAG-Agent  
A clean, modular **RAG pipeline** for answering questions from documents using semantic search and lightweight LLMs.

Includes:
- PDF & web ingestion (PyPDF2, BeautifulSoup)
- Embedding with SentenceTransformers (MiniLM)
- FAISS-based vector search
- QA via Flan-T5 (fast and Colab-compatible)


> Perfect for anyone building semantic search systems or learning how RAG works.

---

### 🤖 Genesis  
An experimental **multi-agent AI research assistant** that reads papers, generates runnable code, and iterates over experiments.

Includes:
- Modular agent architecture (abstracted)
- Architecture generation
- Logging, summaries, and pipeline execution
- Designed for high-level research workflows
- Security-first, offline-compatible. Ideal for vetted collaborators.

> Research automation, reimagined.

---

### 🔥 RAG-Agent-Ops  
A Pinecone-integrated RAG pipeline built for **quick testing and semantic document search**, all inside Colab.

Includes:
- PDF upload with simple Colab widgets
- Text extraction, embedding, and vector storage (Pinecone)
- Natural language querying and semantic retrieval
- Minimal setup, production-ready structure
- HuggingFace all-MiniLM-L6-v2

> A great entry point for learning RAG + Pinecone workflows.

---

## 🧰 Common Technologies

- **LLMs**: OpenRouter (Llama 4 Maverick, Mistral, Qwen), HuggingFace all-MiniLM-L6-v2,  Flan-T5
- **Vector Search**: FAISS, Pinecone, Chroma, Weaviate
- **Embeddings**: SentenceTransformers (MiniLM)
- **Pipelines**: LangGraph, LangChain, LlamaIndex
- **RL & MLOps**: Stable-Baselines3, RLlib, Weights & Biases, MLflow
- **APIs / UIs**: FastAPI, Streamlit, Colab Widgets
- **DevOps / Tooling**: Docker, Kubernetes YAML, Git, GitHub
- **Development**: 100% runnable in Google Colab

---

## 🔀 Repo Strategy

Each project lives in its own Git branch — clean, self-contained, and versionable.

To explore a project:
```bash
git clone https://github.com/ks-suraj/projects
cd projects
git checkout <PROJECT-BRANCH>
