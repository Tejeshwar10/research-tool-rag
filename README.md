# RAG-Based Research Tool

A Python app that lets you ask questions over your own documents  paste in URLs or upload PDFs, and get answers grounded in the actual content with source citations. No hallucinations, no guessing.

Built with LangChain, ChromaDB, HuggingFace embeddings, and LLaMA 3.3-70B via Groq. Frontend is Streamlit.

---

## Why I Built This

Analysts spend way too much time manually skimming through articles and reports to find specific data points. I wanted a tool where you throw in a few sources and just ask your question — and get a cited, accurate answer in seconds.

---

## How It Works

```
User Input (URLs / PDFs)
        │
        ▼
┌─────────────────────┐
│  Document Loader    │  ← WebBaseLoader (URLs) + PyPDFLoader (PDFs)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Text Chunking      │  ← RecursiveCharacterTextSplitter
│  chunk_size = 800   │     chunk_overlap = 100
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Embedding Model    │  ← HuggingFace: Alibaba-NLP/gte-base-en-v1.5
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Vector Store       │  ← ChromaDB (persisted locally)
│  (ChromaDB)         │     Batch ingestion: 50 chunks/batch
└────────┬────────────┘
         │
    Query Time
         │
         ▼
┌─────────────────────┐
│  Retriever          │  ← Top-3 semantically similar chunks
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM                │  ← Groq: LLaMA 3.3-70B-Versatile
│  + RetrievalQA      │     RetrievalQAWithSourcesChain
│  Chain              │
└────────┬────────────┘
         │
         ▼
  Answer + Source Citations
```

1. Documents are split into 800-token chunks with 100-token overlap to preserve context
2. Each chunk is embedded using `Alibaba-NLP/gte-base-en-v1.5` and stored in ChromaDB
3. At query time, the top 3 most relevant chunks are retrieved and passed to the LLM
4. LLaMA generates an answer grounded only in those chunks, with source links included

---

## Tech Stack

- **LLM:** LLaMA 3.3-70B via Groq API
- **Embeddings:** HuggingFace `gte-base-en-v1.5`
- **Vector Store:** ChromaDB (local persistence)
- **Framework:** LangChain `RetrievalQAWithSourcesChain`
- **UI:** Streamlit
- **Loaders:** `WebBaseLoader` for URLs, `PyPDFLoader` for PDFs

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Tejeshwar10/research-tool-rag.git
cd research-tool-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run
streamlit run main.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## Usage

1. Paste up to 3 URLs and/or upload PDFs in the sidebar
2. Hit **Process Sources** — chunks get embedded and indexed
3. Ask anything in the main panel — answers come with source citations

**Example:**
> *"What was the 30-year fixed mortgage rate and when was it reported?"*
> → Answer pulled directly from the CNBC articles with the source URL cited

---

## Project Structure

```
├── main.py          # Streamlit UI
├── rag.py           # RAG pipeline (load, chunk, embed, retrieve, generate)
├── requirements.txt
├── .env             # API keys (not committed)
└── resources/
    ├── uploads/     # Uploaded PDFs
    └── vectorstore/ # ChromaDB index
```

---

## What I'd Add Next

- Retrieval evaluation (precision@k on a test query set)
- Hybrid search (dense + BM25) for better accuracy
- Multi-turn conversation memory
- Support for `.docx` and `.csv` files

---

## Author

**Tejeshwar Singh** — [LinkedIn](https://www.linkedin.com/in/tejeshwarsingh10091991/) 
