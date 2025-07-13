# 🧠 ks-suraj/projects — Autonomous Systems, RAG Pipelines, and AI Agents

Welcome to my **multi-project repository**, where each Git branch contains a standalone AI/ML project — ranging from retrieval-augmented generation (RAG) systems to full-blown multi-agent research engineers and autonomous infrastructure optimizers.

> ⚠️ Each project lives in its own **branch** — not the `main` branch.

---

## 📦 Projects at a Glance

| Branch Name        | Project Title             | Description                                                                 | Technologies |
|--------------------|---------------------------|-----------------------------------------------------------------------------|--------------|
| `AETHER-AI`        | 🌩️ AETHER-AI              | Autonomous cloud infra optimization using LLMs, RAG, and RL                | LangGraph, RLlib, FastAPI, Streamlit, Colab |
| `AutoDocGen`       | 📝 AutoDocGen              | Autonomous Python documentation generator via AST + OpenRouter LLM         | Python, AST, LLM (Llama 4), Colab |
| `AutoRAG-Agent`    | 🔍 AutoRAG-Agent           | Modular GPU-accelerated RAG pipeline for semantic search + LLM answering   | FAISS, LangChain, Transformers, Colab |
| `Genesis`          | 🤖 Genesis                 | Autonomous AI Research Engineer — parses papers and builds runnable ML     | Python, agent framework, modular pipeline |
| `RAG-Agent-Ops`    | 🔥 RAG-Agent-Ops           | Lightweight Pinecone-backed RAG prototype for document QA                  | Pinecone, PyPDF2, Sentence Transformers, Colab |

---

## 🧠 Project Overviews

### 🌩️ [`AETHER-AI`](https://github.com/ks-suraj/projects/tree/AETHER-AI)
A fully autonomous **multi-cloud monitoring and optimization system**. Built using LLMs, RAG, LangGraph DAG orchestration, and simulated reinforcement learning.

**Features:**
- Multi-cloud (AWS, Azure, GCP) support
- RAG + OpenRouter-powered anomaly detection
- RCA agent using LLM + vector store
- Cost optimization via RL
- FastAPI + Streamlit dashboards

> Developed & tested 100% inside **Colab** — deploys via Docker & optional Kubernetes YAMLs.

---

### 📝 [`AutoDocGen`](https://github.com/ks-suraj/projects/tree/AutoDocGen)
An **autonomous Python doc generator** using AST parsing + OpenRouter LLM (Llama 4 Maverick).

**Workflow:**
- Upload Python `.zip` → Extract AST → Send snippets to LLM → Output Markdown docs
- Fully works in Google Colab (no setup)
- Outputs zip of generated documentation

> Great for automating internal code documentation and onboarding.

---

### 🔍 [`AutoRAG-Agent`](https://github.com/ks-suraj/projects/tree/AutoRAG-Agent)
End-to-end **RAG pipeline** for document ingestion and QA — with modular architecture.

**Components:**
- PDF and URL ingestion
- Text chunking via LangChain
- Embedding with MiniLM → FAISS index
- Query → Semantic retrieval → Answer via Flan-T5

> Built for speed and research flexibility. Easily extendable to UI apps.

---

### 🤖 [`Genesis`](https://github.com/ks-suraj/projects/tree/Genesis)
An experimental **multi-agent AI research assistant** that reads papers, generates runnable code, and iterates over experiments.

**Includes:**
- Modular agent architecture (abstracted)
- Architecture generation
- Logging, summaries, and pipeline execution
- Designed for high-level research workflows

> Security-first, offline-compatible. Ideal for vetted collaborators.

---

### 🔥 [`RAG-Agent-Ops`](https://github.com/ks-suraj/projects/tree/RAG-Agent-Ops)
Lean & functional **Colab-based RAG pipeline** optimized for Pinecone vector DB.

**Pipeline Flow:**
- Upload PDF → Chunk → Embed → Store in Pinecone
- Query via natural language → Retrieve matching chunks
- Built-in Colab widget interface (no external servers needed)

---

## 🔎 How to Explore

Each project lives in its **own Git branch**.  
To check one out:

```bash
git clone https://github.com/ks-suraj/projects.git
cd projects
git checkout AETHER-AI  # or any branch listed above
