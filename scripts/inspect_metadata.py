# -*- coding: utf-8 -*-
"""
Inspect metadata yang sudah di-rebuild dengan enriched metadata
"""
import pickle
import os

print("=" * 70)
print("INSPEKSI METADATA BARU (AFTER REBUILD)")
print("=" * 70)

meta_path = os.path.join("faiss_store", "metadata.pkl")

with open(meta_path, "rb") as f:
    metadata = pickle.load(f)

print(f"\nTotal chunks: {len(metadata)}")

print("\n" + "=" * 70)
print("METADATA STRUCTURE")
print("=" * 70)

first_meta = metadata[0]
print(f"\nAvailable keys: {list(first_meta.keys())}")
print(f"Total keys: {len(first_meta.keys())}")

print("\n" + "=" * 70)
print("CONTOH 3 CHUNKS DENGAN METADATA LENGKAP")
print("=" * 70)

for i, meta in enumerate(metadata[:3]):
    print(f"\n{'='*70}")
    print(f"CHUNK {i+1}")
    print('='*70)
    
    print(f"\nBasic Info:")
    print(f"  chunk_id: {meta['chunk_id']}")
    print(f"  chunk_size: {meta['chunk_size']} chars")
    print(f"  text preview: {meta['text'][:80]}...")
    
    print(f"\nSource Tracking:")
    print(f"  source: {meta['source']}")
    print(f"  page: {meta['page']}")
    print(f"  page_label: {meta['page_label']}")
    print(f"  total_pages: {meta['total_pages']}")
    
    print(f"\nDocument Metadata:")
    print(f"  doc_author: {meta['doc_author']}")
    print(f"  doc_title: {meta['doc_title']}")
    print(f"  creation_date: {meta['creation_date']}")
    print(f"  creator: {meta['creator']}")
    
    print(f"\nOriginal Metadata:")
    print(f"  Keys: {list(meta['original_metadata'].keys())}")

print("\n" + "=" * 70)
print("STATISTIK SOURCES")
print("=" * 70)

sources = set(m['source'] for m in metadata)
print(f"\nUnique source files: {len(sources)}")
for source in sorted(sources):
    count = sum(1 for m in metadata if m['source'] == source)
    print(f"  {source}: {count} chunks")

print("\n" + "=" * 70)
print("COMPARISON: BEFORE vs AFTER")
print("=" * 70)

print("\nBEFORE REBUILD:")
print("  Keys: ['text']")
print("  Total: 1 key")

print("\nAFTER REBUILD:")
print(f"  Keys: {list(first_meta.keys())}")
print(f"  Total: {len(first_meta.keys())} keys")

print("\nIMPROVEMENT:")
print(f"  +{len(first_meta.keys()) - 1} additional metadata fields!")

print("\n" + "=" * 70)
print("SELESAI - Metadata sudah lengkap!")
print("=" * 70)
