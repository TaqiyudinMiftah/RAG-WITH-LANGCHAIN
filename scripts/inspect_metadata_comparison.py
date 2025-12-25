# -*- coding: utf-8 -*-
"""
Demonstrasi: Metadata Minimal vs Metadata Lengkap
"""
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

print("=" * 70)
print("PERBANDINGAN: METADATA MINIMAL VS LENGKAP")
print("=" * 70)

# Load dokumen
docs = load_all_documents("data")
print(f"\nLoaded {len(docs)} documents")

# Ambil 1 dokumen untuk demo
demo_doc = docs[0]

print("\n" + "=" * 70)
print("ORIGINAL DOCUMENT METADATA")
print("=" * 70)
print("\nMetadata dari dokumen asli:")
for key, value in demo_doc.metadata.items():
    if isinstance(value, str) and len(value) > 50:
        print(f"  {key}: {value[:50]}...")
    else:
        print(f"  {key}: {value}")

# Chunking
pipeline = EmbeddingPipeline(chunk_size=1000, chunk_overlap=50)
chunks = pipeline.chunk_documents([demo_doc])

print(f"\n{len([demo_doc])} dokumen -> {len(chunks)} chunks")

print("\n" + "=" * 70)
print("APPROACH 1: METADATA MINIMAL (Current)")
print("=" * 70)

# Current approach - hanya text
minimal_metadata = [{"text": chunk.page_content} for chunk in chunks]

print("\nContoh metadata chunk pertama:")
print(minimal_metadata[0])
print(f"\nKeys: {list(minimal_metadata[0].keys())}")
print("Problem: Kehilangan info source, page, dll!")

print("\n" + "=" * 70)
print("APPROACH 2: METADATA LENGKAP (Improved)")
print("=" * 70)

# Improved approach - preserve original metadata + add chunk info
enriched_metadata = []
for i, chunk in enumerate(chunks):
    meta = {
        "text": chunk.page_content,
        "chunk_id": i,
        "chunk_size": len(chunk.page_content),
        "source": chunk.metadata.get("source", "unknown"),
        "page": chunk.metadata.get("page", None),
        "total_pages": chunk.metadata.get("total_pages", None),
        # Preserve all original metadata
        "original_metadata": chunk.metadata
    }
    enriched_metadata.append(meta)

print("\nContoh metadata chunk pertama (enriched):")
for key, value in enriched_metadata[0].items():
    if key == "text":
        print(f"  {key}: {value[:80]}...")
    elif key == "original_metadata":
        print(f"  {key}: <dict with {len(value)} keys>")
    else:
        print(f"  {key}: {value}")

print(f"\nKeys: {list(enriched_metadata[0].keys())}")

print("\n" + "=" * 70)
print("BENEFIT METADATA LENGKAP")
print("=" * 70)

print("\nDengan metadata lengkap, Anda bisa:")
print("1. Filter hasil by source file")
print("2. Track chunk berasal dari halaman mana")
print("3. Display context yang lebih baik ke user")
print("4. Debug issues lebih mudah")
print("5. Implement advanced features (re-ranking, filtering, etc)")

print("\n" + "=" * 70)
print("COMPARISON SIZE")
print("=" * 70)

import pickle
import sys

minimal_size = sys.getsizeof(pickle.dumps(minimal_metadata[:10]))
enriched_size = sys.getsizeof(pickle.dumps(enriched_metadata[:10]))

print(f"\n10 chunks metadata size:")
print(f"  Minimal:  {minimal_size:,} bytes")
print(f"  Enriched: {enriched_size:,} bytes")
print(f"  Overhead: {enriched_size - minimal_size:,} bytes ({((enriched_size/minimal_size - 1) * 100):.1f}%)")

print("\nKesimpulan: Overhead kecil, tapi benefit besar!")

print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)
