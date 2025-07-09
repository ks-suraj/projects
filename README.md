# 🔥 RAG-Agent-Ops

A **Retrieval-Augmented Generation (RAG)** pipeline designed for seamless document ingestion, vector storage, and semantic search — optimized for Colab & Pinecone integration.

> 🚀 **Try the full working pipeline in Colab here:**  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18LWSgw0Q_f0Rh_A1da5rDGB9MTzpFBVF#scrollTo=Rk1SGfgSN2_4)

---

## 🏗️ Project Structure

  📂 Project Structure — RAG-Agent-Ops
  
  ragagentops/                 // Root project directory
  
    ├── app/                  // App-specific modules
    │   └── services/         // Service layers (agents, DB handlers)
    │       ├── embedding_agent.py    // Embedding agent using Sentence Transformers
    │       └── vector_store.py       // Pinecone vector DB interface
    
    ├── ui_widgets/           // (Optional) Colab Widgets for UI (if added later)
    
    ├── main_pipeline.ipynb   // Colab notebook pipeline (interactive, end-to-end)
    
    └── README.md             // Project documentation (this file)

---

## 📋 Workflow & Pipeline Flow

1. **Upload PDF Document**  
2. **Extract Text & Chunk**  
3. **Generate Embeddings** (via `sentence-transformers/all-MiniLM-L6-v2`)  
4. **Store Vectors in Pinecone**  
5. **Query via Natural Language & Retrieve Relevant Chunks**

---

## 🧩 Technologies Used
- **Python 3**
- **Sentence Transformers** (for text embedding)
- **Pinecone** (for vector storage and search)
- **PyPDF2** (for PDF parsing)
- **Colab Widgets / IPython Widgets** (for minimal UI)

---

## 📦 Key Libraries
| Library           | Purpose                         |
|-------------------|---------------------------------|
| `sentence-transformers` | Embedding text into vectors |
| `pinecone`        | Vector DB storage & search      |
| `PyPDF2`          | PDF text extraction             |
| `ipywidgets`      | Interactive UI in Colab         |

---

## 📋 Colab Notebook Highlights:
- ✅ No API or UI dependencies at first; works directly in Colab cells  
- ✅ Pinecone API key fetched securely from Colab Secrets  
- ✅ Simple Widget-based interface for PDF Upload, Document Ingestion & Querying  
- ✅ Easily extendable to APIs or Gradio later  

---

## 📂 Deployment Notes
- Runs entirely on **Colab Free Tier (CPU)**
- Pinecone index must already exist before running  
- No external servers needed.

---

## 💡 Why This Project?
This pipeline provides an **end-to-end working prototype** for building semantic search solutions — ideal for learning **Vector DBs + RAG Pipelines** or adapting for production systems.

---

## 📎 Colab Notebook (with Full Code & Demo):
➡️ [Open in Colab → Complete Working Pipeline](https://colab.research.google.com/drive/18LWSgw0Q_f0Rh_A1da5rDGB9MTzpFBVF#scrollTo=Rk1SGfgSN2_4)

---

## 👨‍💻 Author
**@ks-suraj**  
*(Maintainer of this project)*

---

