# 📚 RAG untuk Review Paper Terbaru - Complete Guide

## 🎯 Use Case Anda: PERFECT untuk RAG!

**Problem:**
- LLM (GPT-4, Claude, dll) tidak tahu tentang paper terbaru
- Paper baru keluar setiap hari
- Anda ingin bisa tanya tentang paper tanpa manual summarize

**Solution dengan RAG:**
- Upload PDF paper → Otomatis masuk vector database
- Tanya apa saja tentang paper
- Sistem retrieve relevant parts + LLM generate answer
- Dapat jawaban dengan citations!

---

## ✅ Sistem Anda SUDAH SIAP 80%!

### Yang Sudah Ada:
- ✅ PDF Loader (PyPDFLoader)
- ✅ Chunking (RecursiveCharacterTextSplitter)
- ✅ Embedding (all-MiniLM-L6-v2)
- ✅ Vector Store (FAISS)
- ✅ Retrieval (Similarity Search)
- ✅ Metadata Lengkap (source, page, author, date)

### Yang Perlu Ditambahkan:
- ⚠️ **LLM Integration** (untuk generate answer)

---

## 🚀 Quick Start: 3 Langkah Sederhana

### 1. Upload Paper Baru

```python
from paper_review_rag import PaperReviewRAG

# Initialize
rag = PaperReviewRAG("faiss_store")

# Upload paper baru
rag.add_new_paper("data/pdf/your_new_paper.pdf")
```

**Apa yang terjadi:**
- PDF di-load dan split per halaman
- Setiap halaman di-chunk jadi potongan kecil
- Setiap chunk di-embed jadi vector
- Vector disimpan di FAISS index
- Metadata lengkap disimpan (source, page, etc)

### 2. Tanya Tentang Paper

```python
# Tanpa LLM (hanya retrieval)
result = rag.ask_about_paper("What is the main contribution?", top_k=5)

# Dengan LLM (retrieval + generation)
from rag_with_llm import RAGWithLLM
rag_llm = RAGWithLLM("faiss_store", llm_provider="ollama")
result = rag_llm.query("What is the main contribution?")
print(result['answer'])
```

### 3. Lihat Hasil dengan Citations

```python
print(f"Answer: {result['answer']}")
print("\nSources:")
for src in result['sources']:
    print(f"  - {src['source']} (page {src['page']})")
```

---

## 💡 Pilihan LLM (Generation Part)

### Option 1: Ollama (Local LLM) - **RECOMMENDED untuk Start**

**Keuntungan:**
- ✅ **GRATIS 100%**
- ✅ **Data tetap lokal** (privacy terjaga)
- ✅ **Tidak perlu API key**
- ✅ **Tidak ada rate limit**
- ✅ **Bisa offline**

**Setup:**
```bash
# 1. Install Ollama
# Download dari: https://ollama.ai

# 2. Pull model (pilih salah satu)
ollama pull llama2        # 7B - Good balance (3.8 GB)
ollama pull mistral       # 7B - Fast & capable (4.1 GB)
ollama pull llama2:13b    # 13B - Better quality (7.3 GB)

# 3. Install Python package
pip install ollama

# 4. Jalankan RAG
python rag_with_llm.py
```

**Contoh Code:**
```python
from rag_with_llm import RAGWithLLM

rag = RAGWithLLM("faiss_store", llm_provider="ollama")
result = rag.query("What is the methodology in this paper?")
print(result['answer'])
```

---

### Option 2: OpenAI (GPT-3.5/GPT-4)

**Keuntungan:**
- ✅ Kualitas jawaban terbaik
- ✅ Setup mudah
- ✅ Cepat

**Kekurangan:**
- ❌ Berbayar (~$0.002 per query untuk GPT-3.5)
- ❌ Data terkirim ke OpenAI
- ❌ Perlu API key

**Setup:**
```bash
# 1. Get API key dari: https://platform.openai.com/api-keys
# 2. Install package
pip install openai

# 3. Set API key
export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
set OPENAI_API_KEY=your-api-key-here       # Windows

# 4. Jalankan
python rag_with_llm.py
```

---

### Option 3: HuggingFace API

**Keuntungan:**
- ✅ FREE tier tersedia
- ✅ Banyak pilihan model

**Kekurangan:**
- ❌ Rate limit pada free tier
- ❌ Data terkirim ke HuggingFace

**Setup:**
```bash
pip install huggingface_hub
export HUGGINGFACE_API_KEY="your-hf-token"
```

---

## 📝 Complete Workflow Example

### Scenario: Review Paper Terbaru tentang Climate Change

```python
from paper_review_rag import PaperReviewRAG
from rag_with_llm import RAGWithLLM

# Step 1: Initialize system
rag = PaperReviewRAG("faiss_store")

# Step 2: Upload paper baru yang baru keluar
print("Uploading new paper...")
rag.add_new_paper("data/pdf/climate_paper_2024.pdf")

# Step 3: Lihat apa saja paper yang ada
rag.list_papers()

# Step 4: Tanya tentang paper dengan LLM
rag_llm = RAGWithLLM("faiss_store", llm_provider="ollama")

questions = [
    "What is the main research question of this paper?",
    "What methodology did the authors use?",
    "What are the key findings?",
    "What are the limitations mentioned?",
    "How does this compare to previous work?"
]

for question in questions:
    print(f"\nQ: {question}")
    result = rag_llm.query(question, top_k=5)
    print(f"A: {result['answer']}")
    print(f"Sources: {', '.join([s['source'] for s in result['sources']])}")
```

---

## 🔥 Advanced Features

### 1. Query Specific Paper Only

```python
# Hanya search dalam paper tertentu
result = rag.ask_about_paper(
    "What is the dataset used?",
    top_k=3,
    source_filter="climate_paper_2024.pdf"
)
```

### 2. Compare Multiple Papers

```python
question = "How do these papers define sentiment analysis?"
result = rag_llm.query(question, top_k=10)
# LLM akan compare dari semua papers
```

### 3. Filter by Date

```python
# Get papers after certain date
recent_papers = [m for m in rag.store.metadata 
                 if m.get('creation_date', '') > '2024-01-01']
```

### 4. Track Paper Statistics

```python
from collections import Counter

# Chunks per paper
sources = [m['source'] for m in rag.store.metadata]
stats = Counter(sources)

for paper, count in stats.most_common():
    print(f"{paper}: {count} chunks")
```

---

## 🎓 Best Practices untuk Paper Review

### 1. Struktur Pertanyaan yang Baik

**Good Questions:**
- "What is the main contribution of this paper?"
- "What dataset and evaluation metrics were used?"
- "What are the limitations mentioned by the authors?"
- "How does this approach differ from previous work?"

**Avoid:**
- "Is this good?" (subjective, LLM akan bias)
- "Tell me everything" (terlalu broad)

### 2. Verify dengan Citations

Selalu check sources yang dikembalikan:
```python
result = rag_llm.query("What is the accuracy?")
print("\nSources used:")
for src in result['sources']:
    print(f"  {src['source']} - Page {src['page']}")
```

### 3. Chunk Size Optimization

Untuk paper teknis, consider:
- Increase chunk_size untuk context lebih panjang
- Adjust overlap untuk tidak putus di tengah formula

```python
store = FaissVectorStore(
    chunk_size=1500,  # Lebih besar untuk paper
    chunk_overlap=100  # Overlap lebih besar
)
```

---

## 📊 Cost Comparison

| Method | Setup Time | Cost per Query | Privacy | Quality |
|--------|-----------|---------------|---------|---------|
| **Ollama (Local)** | 15 min | FREE | 100% Private | Good |
| **OpenAI GPT-3.5** | 5 min | ~$0.002 | Sent to OpenAI | Excellent |
| **OpenAI GPT-4** | 5 min | ~$0.03 | Sent to OpenAI | Best |
| **HuggingFace** | 10 min | FREE (limited) | Sent to HF | Good |

**Recommendation:**
- **For learning/testing:** Ollama (free, private)
- **For production (few users):** OpenAI GPT-3.5 (cheap, reliable)
- **For production (many users):** Ollama on server (free, scalable)

---

## 🐛 Troubleshooting

### "Model doesn't know about the paper"
- ✅ Check: Paper sudah di-upload? Run `rag.list_papers()`
- ✅ Check: Query menggunakan RAG? Bukan direct LLM call

### "Retrieval returns wrong chunks"
- ✅ Try: Increase top_k (retrieve more chunks)
- ✅ Try: Rephrase question (use keywords from paper)
- ✅ Check: Embedding model cocok untuk domain paper

### "Ollama error: connection refused"
- ✅ Check: Ollama running? Start with `ollama serve`
- ✅ Check: Model downloaded? Run `ollama list`

### "Answer doesn't match paper content"
- ✅ Check sources: Verify chunks yang di-retrieve relevan
- ✅ Try: Adjust prompt untuk lebih specific
- ✅ Consider: Use better LLM (GPT-4 vs GPT-3.5)

---

## 🎯 Summary

**Anda SUDAH PUNYA sistem yang bagus!** Tinggal tambah LLM:

1. **Upload PDF paper** → `rag.add_new_paper()`
2. **Ask questions** → `rag_llm.query()`
3. **Get answers with citations** → Done! ✅

**Next Steps:**
1. Install Ollama (5 minutes)
2. Run `python rag_with_llm.py`
3. Start reviewing papers! 🚀

---

## 📚 Files Reference

- `paper_review_rag.py` - Main RAG system (retrieval)
- `rag_with_llm.py` - LLM integration (generation)
- `src/vectorstore.py` - Vector store implementation
- `src/data_loader.py` - PDF loading
- `src/embedding.py` - Embedding pipeline

**Try it now:**
```bash
python paper_review_rag.py  # See retrieval in action
python rag_with_llm.py      # Full RAG with LLM
```
