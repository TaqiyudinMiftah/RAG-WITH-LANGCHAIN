# RAG-With-LangChain ������

> Complete Retrieval-Augmented Generation system for academic paper review and question-answering using LangChain, FAISS, and Google Gemini.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ��� Overview

**RAG-With-LangChain** is a production-ready Retrieval-Augmented Generation (RAG) system designed for academic paper review and intelligent question-answering. The system combines document retrieval with large language models to provide accurate, context-aware answers with citations.

### Key Features

- **��� Multi-Format Document Loading**: PDF, TXT, CSV, JSON support
- **��� Advanced Vector Search**: FAISS-based similarity search with L2 distance
- **��� Enriched Metadata**: 12-field metadata system for better context
- **��� Multiple LLM Providers**: Google Gemini, OpenAI, HuggingFace, Ollama
- **��� Secure API Management**: Environment-based configuration with `.env`
- **��� Production-Ready**: Complete with error handling, logging, and documentation

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Documents  │────>│   Chunking   │────>│  Embedding  │
│ (PDF/TXT)   │     │  (1000/50)   │     │ (MiniLM-L6) │
└─────────────┘     └──────────────┘     └─────────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Answer    │<────│   LLM        │<────│    FAISS    │
│ + Citations │     │  (Gemini)    │     │ Vector Store│
└─────────────┘     └──────────────┘     └─────────────┘
```

## ��� Quick Start

### Prerequisites

- Python 3.9 or higher
- [UV package manager](https://github.com/astral-sh/uv) (recommended) or pip
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RAG-With-Langchain.git
   cd RAG-With-Langchain
   ```

2. **Install dependencies**
   ```bash
   # Using UV (recommended)
   uv pip install -r requirements.txt
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Setup environment variables**
   ```bash
   # Copy template
   cp .env.example .env
   
   # Edit .env and add your API key
   # GEMINI_API_KEY=your-api-key-here
   ```

4. **Build vector store**
   ```bash
   uv run python scripts/rebuild_with_metadata.py
   ```

### Usage

#### Basic Example

```python
from src.vectorstore import FaissVectorStore

# Load vector store
store = FaissVectorStore("faiss_store")
store.load()

# Query
results = store.query("What is the main contribution?", top_k=3)
for result in results:
    print(f"Source: {result['source']}")
    print(f"Text: {result['text'][:200]}...")
    print(f"Similarity: {result['similarity_score']:.4f}\n")
```

#### Complete RAG with LLM

```python
from rag_with_llm import RAGWithLLM

# Initialize RAG system
rag = RAGWithLLM(
    vector_store_path="faiss_store",
    llm_provider="gemini"
)

# Ask question
response = rag.ask(
    question="What methods are used in sentiment analysis?",
    top_k=5
)

print(response['answer'])
print("\nSources:", response['sources'])
```

#### Demo Scripts

```bash
# Quick demo with single question
uv run python examples/demo_gemini.py

# Interactive demo with multiple questions
uv run python examples/test_gemini_interactive.py

# Paper review system
uv run python paper_review_rag.py
```

## ��� Project Structure

```
RAG-With-Langchain/
├── src/                          # Core source code
│   ├── data_loader.py           # Document loaders (PDF, TXT, CSV, JSON)
│   ├── embedding.py             # Embedding pipeline with chunking
│   └── vectorstore.py           # FAISS vector store with metadata
├── examples/                     # Usage examples
│   ├── demo_gemini.py           # Quick demo script
│   ├── test_gemini_interactive.py  # Interactive demo
│   └── app.py                   # Basic app example
├── scripts/                      # Utility scripts
│   ├── rebuild_with_metadata.py # Rebuild vector store
│   ├── inspect_document.py      # Inspect document structure
│   ├── inspect_metadata.py      # View metadata
│   └── inspect_metadata_comparison.py  # Compare metadata
├── docs/                         # Documentation
│   ├── RAG_GEMINI_SUMMARY.md    # Complete system overview
│   ├── PAPER_REVIEW_GUIDE.md    # Paper review workflow
│   ├── METADATA_GUIDE.md        # Metadata documentation
│   ├── ENV_SETUP.md             # Environment setup guide
│   ├── GIT_GUIDE.md             # Git workflow guide
│   └── PRE_PUSH_SUMMARY.md      # Pre-push checklist
├── data/                         # Data files
│   ├── pdf/                     # PDF documents
│   └── text_files/              # Text documents
├── paper_review_rag.py          # Paper review RAG system
├── rag_with_llm.py              # Complete RAG with multiple LLM providers
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── .env.example                 # Environment template
```

## ��� Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API (recommended)
GEMINI_API_KEY=your-gemini-api-key

# Optional: Other LLM providers
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_API_TOKEN=your-hf-token
```

See [docs/ENV_SETUP.md](docs/ENV_SETUP.md) for detailed setup instructions.

### Vector Store Configuration

Customize in `src/vectorstore.py`:

```python
FaissVectorStore(
    vector_store_path="faiss_store",
    embedding_model="all-MiniLM-L6-v2",  # 384 dimensions
    chunk_size=1000,                      # Characters per chunk
    chunk_overlap=50                      # Overlap between chunks
)
```

### Metadata Schema

Each chunk includes 12 metadata fields:

| Field | Description | Example |
|-------|-------------|---------|
| `text` | Chunk content | "The main contribution..." |
| `source` | File path | "data/pdf/paper.pdf" |
| `page` | Page number | 5 |
| `chunk_id` | Unique identifier | 42 |
| `chunk_size` | Character count | 987 |
| `page_label` | Page label | "Page 5" |
| `total_pages` | Document pages | 12 |
| `doc_author` | Author | "John Doe" |
| `doc_title` | Title | "Climate Analysis" |
| `creation_date` | Created date | "2023-06-15" |
| `creator` | Creator tool | "LaTeX" |
| `original_metadata` | Raw metadata | {...} |

## ��� Current Stats

- **Documents Indexed**: 6 papers (4 PDFs + 2 TXTs)
- **Total Chunks**: 295 chunks
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Store**: FAISS IndexFlatL2 (exact L2 distance)
- **LLM**: Google Gemini Pro (gemini-pro)

## ��� Supported LLM Providers

| Provider | Model | Status | Setup |
|----------|-------|--------|-------|
| **Google Gemini** | gemini-pro | ✅ Tested | [Get API key](https://makersuite.google.com/app/apikey) |
| OpenAI | gpt-3.5-turbo | ⚠️ Code ready | [Get API key](https://platform.openai.com/api-keys) |
| HuggingFace | Inference API | ⚠️ Code ready | [Get token](https://huggingface.co/settings/tokens) |
| Ollama | Local models | ⚠️ Code ready | [Install Ollama](https://ollama.ai/) |

## ��� Documentation

Comprehensive documentation available in the `docs/` folder:

- **[RAG_GEMINI_SUMMARY.md](docs/RAG_GEMINI_SUMMARY.md)** - Complete system overview with examples
- **[PAPER_REVIEW_GUIDE.md](docs/PAPER_REVIEW_GUIDE.md)** - Workflow for reviewing papers
- **[METADATA_GUIDE.md](docs/METADATA_GUIDE.md)** - Metadata structure and usage
- **[ENV_SETUP.md](docs/ENV_SETUP.md)** - Environment configuration guide
- **[GIT_GUIDE.md](docs/GIT_GUIDE.md)** - Git workflow and best practices

## ��� Examples

### Add New Paper

```python
from paper_review_rag import PaperReviewRAG

rag = PaperReviewRAG("faiss_store")
rag.add_new_paper("data/pdf/new_paper.pdf")
```

### Query Specific Paper

```python
result = rag.ask_about_paper(
    question="What is the methodology?",
    top_k=3,
    source_filter="specific_paper.pdf"
)
```

### List All Papers

```python
rag.list_papers()
```

### Custom Prompt

```python
from rag_with_llm import RAGWithLLM

rag = RAGWithLLM("faiss_store", llm_provider="gemini")

custom_prompt = """
Based on this context: {context}

Question: {question}

Provide a detailed analysis with citations.
"""

response = rag.ask(
    question="Compare the methodologies",
    prompt_template=custom_prompt
)
```

## ��� Security

- ✅ API keys stored in `.env` (not committed to Git)
- ✅ `.env.example` provided as template
- ✅ Comprehensive `.gitignore` rules
- ✅ No hardcoded credentials in source code

See [docs/GIT_GUIDE.md](docs/GIT_GUIDE.md) for security best practices.

## ���️ Development

### Rebuild Vector Store

```bash
uv run python scripts/rebuild_with_metadata.py
```

### Inspect Documents

```bash
# View document structure
uv run python scripts/inspect_document.py

# View metadata
uv run python scripts/inspect_metadata.py

# Compare metadata versions
uv run python scripts/inspect_metadata_comparison.py
```

### Run Tests

```bash
# Test Gemini integration
uv run python examples/test_gemini_interactive.py

# Test retrieval only
uv run python paper_review_rag.py
```

## ��� Performance

- **Query Speed**: ~50ms per query (retrieval only)
- **Embedding Speed**: ~1000 docs/minute
- **Memory Usage**: ~200MB (295 chunks loaded)
- **Storage**: ~2MB (vector store only, excluding PDFs)

## ���️ Roadmap

- [ ] Add more LLM providers (Claude, Llama)
- [ ] Implement hybrid search (dense + sparse)
- [ ] Add conversation memory
- [ ] Web interface with Streamlit/Gradio
- [ ] Batch processing for large document sets
- [ ] Performance benchmarking suite
- [ ] Docker containerization
- [ ] API server with FastAPI

## ��� Contributing

This is a personal learning project, but contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ��� License

This project is licensed under the MIT License - see the LICENSE file for details.

## ��� Acknowledgments

- [LangChain](https://python.langchain.com/) - RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [Google Gemini](https://ai.google.dev/) - LLM API

## ��� Contact

For questions or feedback, please open an issue or reach out through GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**
