# Metadata dalam Vector Store - Dokumentasi

## 📋 Current State (vectorstore.py)

### Metadata yang Tersimpan Sekarang

**Struktur:**
```python
{
    "text": "chunk content..."
}
```

**Statistik (dari faiss_store/metadata.pkl):**
- Total entries: 295 chunks
- Keys: Hanya 1 (`text`)
- Text length: 76-999 karakter (rata-rata 870 chars)

**Kode yang Menyebabkan Ini (vectorstore.py line 26):**
```python
metadatas = [{"text": chunk.page_content} for chunk in chunks]
```

### Masalah dengan Approach Ini

❌ **Informasi Hilang:**
- Source file (tidak tahu chunk dari file mana)
- Page number (PDF)
- Document metadata (author, title, date, dll)

❌ **Impact:**
- Sulit debugging
- Tidak bisa implement filtering
- Tidak bisa show citations ke user
- Tidak bisa track provenance

---

## ✅ Improved Approach (improved_vectorstore.py)

### Metadata yang Seharusnya Tersimpan

**Struktur Lengkap:**
```python
{
    # Basic chunk info
    "text": "chunk content...",
    "chunk_id": 0,
    "chunk_size": 979,
    
    # Source tracking
    "source": "data\\pdf\\2211.03533v1.pdf",
    
    # PDF-specific
    "page": 0,
    "page_label": "1",
    "total_pages": 12,
    
    # Document metadata
    "doc_author": "",
    "doc_title": "",
    "creation_date": "2022-11-08T01:54:42+00:00",
    
    # Original metadata (complete)
    "original_metadata": {
        "producer": "pdfTeX-1.40.21",
        "creator": "LaTeX with hyperref",
        ...
    }
}
```

### Keuntungan

✅ **Complete Tracking:**
- Tahu persis chunk dari mana
- Bisa trace kembali ke dokumen asli

✅ **Advanced Features:**
- Filter by source: `filter_by_source(results, "paper.pdf")`
- Show citations: "Found in paper.pdf, page 3"
- Implement re-ranking based on source quality

✅ **Better UX:**
- User bisa lihat dari mana informasi berasal
- Trust & transparency

✅ **Debugging:**
- Mudah identify problematic chunks
- Better logging

---

## 🔄 Migration Path

### Option 1: Rebuild (Recommended)
```python
from improved_vectorstore import ImprovedFaissVectorStore
from src.data_loader import load_all_documents

# Load documents
docs = load_all_documents("data")

# Build dengan improved version
store = ImprovedFaissVectorStore("faiss_store_v2")
store.build_from_documents(docs)

# Query
results = store.query("What is LangChain?", top_k=3)

# Sekarang results punya metadata lengkap!
for result in results:
    print(f"Source: {result['source']}")
    print(f"Page: {result['page']}")
    print(f"Text: {result['text'][:100]}...")
```

### Option 2: Update Existing Code
Update `src/vectorstore.py` baris 26 dari:
```python
metadatas = [{"text": chunk.page_content} for chunk in chunks]
```

Menjadi:
```python
metadatas = []
for i, chunk in enumerate(chunks):
    meta = {
        "text": chunk.page_content,
        "chunk_id": i,
        "source": chunk.metadata.get("source", "unknown"),
        "page": chunk.metadata.get("page"),
        # Add more fields as needed
    }
    metadatas.append(meta)
```

---

## 📊 Memory Overhead

**Perbandingan:**

**Minimal metadata (current):**
```
1 chunk = ~50 bytes metadata (just text reference)
295 chunks = ~15 KB
```

**Enriched metadata (improved):**
```
1 chunk = ~200 bytes metadata (complete info)
295 chunks = ~59 KB
```

**Overhead:** ~44 KB tambahan (0.04 MB)

**Kesimpulan:** Overhead sangat kecil (~300% increase), tapi benefit sangat besar!

---

## 🎯 Best Practices

### Metadata Minimal yang Wajib Ada

```python
{
    "text": "...",           # Wajib: content chunk
    "source": "...",         # Wajib: source file
    "chunk_id": 0,           # Recommended: tracking
}
```

### Metadata Optional tapi Berguna

```python
{
    "page": 0,              # Untuk PDF/DOCX
    "page_label": "1",      # Human-readable page
    "section": "...",       # Untuk dokumen berstruktur
    "timestamp": "...",     # Kapan di-index
    "doc_type": "pdf",      # Type dokumen
    "language": "en",       # Bahasa
}
```

### Metadata untuk Production

```python
{
    # ... basic fields ...
    "indexed_at": "2024-12-25T10:30:00",
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_method": "RecursiveCharacterTextSplitter",
    "version": "1.0",       # Schema version untuk migration
    "hash": "abc123...",    # Content hash untuk deduplication
}
```

---

## 🚀 Next Steps

1. **Review** improved_vectorstore.py
2. **Decide** apakah mau rebuild index atau update code
3. **Test** dengan sample queries
4. **Compare** hasil search dengan metadata lengkap
5. **Implement** advanced features (filtering, citations, dll)

---

## 📚 References

- Original code: `src/vectorstore.py`
- Improved code: `improved_vectorstore.py`
- Current metadata: `faiss_store/metadata.pkl`
- Documentation: This file
