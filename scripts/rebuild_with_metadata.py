# -*- coding: utf-8 -*-
"""
Script untuk rebuild vector store dengan metadata lengkap
"""
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
import os
import shutil

print("=" * 70)
print("REBUILD VECTOR STORE DENGAN METADATA LENGKAP")
print("=" * 70)

# Backup existing vector store
if os.path.exists("faiss_store"):
    print("\n[BACKUP] Backing up existing vector store...")
    if os.path.exists("faiss_store_backup"):
        shutil.rmtree("faiss_store_backup")
    shutil.copytree("faiss_store", "faiss_store_backup")
    print("[BACKUP] Backup saved to: faiss_store_backup/")

print("\n[STEP 1] Loading documents...")
docs = load_all_documents("data")
print(f"[INFO] Loaded {len(docs)} documents")

# Show sample metadata
if len(docs) > 0:
    print("\n[INFO] Sample document metadata:")
    sample = docs[0]
    for key, value in list(sample.metadata.items())[:5]:
        if isinstance(value, str) and len(value) > 50:
            print(f"  {key}: {value[:50]}...")
        else:
            print(f"  {key}: {value}")

print("\n[STEP 2] Building vector store with enriched metadata...")
print("[INFO] This may take a few minutes depending on document size...")

store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)

print("\n[STEP 3] Verifying build...")
store.load()

# Get stats
total_chunks = len(store.metadata)
print(f"[SUCCESS] Vector store rebuilt successfully!")
print(f"[INFO] Total chunks indexed: {total_chunks}")

# Sample metadata check
if total_chunks > 0:
    print("\n[INFO] Sample enriched metadata (first chunk):")
    sample_meta = store.metadata[0]
    for key, value in sample_meta.items():
        if key == "text":
            print(f"  {key}: {value[:80]}...")
        elif key == "original_metadata":
            print(f"  {key}: <dict with {len(value)} keys>")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n[INFO] Metadata keys available: {list(sample_meta.keys())}")

print("\n" + "=" * 70)
print("REBUILD COMPLETED")
print("=" * 70)
print("\nNext steps:")
print("1. Test query dengan: python app.py")
print("2. Inspect metadata dengan: python inspect_metadata.py")
print("3. Backup tersimpan di: faiss_store_backup/")
