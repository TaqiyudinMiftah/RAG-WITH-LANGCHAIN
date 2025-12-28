# -*- coding: utf-8 -*-
"""
Script untuk melihat chunks di vector store

Usage:
    # Lihat semua chunks (first 10)
    python scripts/view_chunks.py
    
    # Lihat chunk tertentu
    python scripts/view_chunks.py --chunk_id 5
    
    # Filter by source file
    python scripts/view_chunks.py --source climate_paper.pdf
    
    # Search keyword dalam chunks
    python scripts/view_chunks.py --search "sentiment analysis"
    
    # Lihat chunks dengan limit
    python scripts/view_chunks.py --limit 20
"""

import sys
import os
from pathlib import Path
import argparse
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore import FaissVectorStore


def print_chunk(chunk_meta: dict, show_full_text: bool = False):
    """Print chunk information dengan format yang rapi"""
    print("\n" + "="*80)
    print(f"📄 CHUNK #{chunk_meta.get('chunk_id', 'N/A')}")
    print("="*80)
    
    # Basic info
    print(f"Source      : {chunk_meta.get('source', 'N/A')}")
    print(f"Page        : {chunk_meta.get('page_label', 'N/A')}")
    print(f"Chunk Size  : {chunk_meta.get('chunk_size', 0)} characters")
    
    # Document metadata
    if chunk_meta.get('doc_title'):
        print(f"Doc Title   : {chunk_meta['doc_title']}")
    if chunk_meta.get('doc_author'):
        print(f"Doc Author  : {chunk_meta['doc_author']}")
    
    # Text content
    text = chunk_meta.get('text', '')
    print(f"\n📝 Content ({len(text)} chars):")
    print("-"*80)
    
    if show_full_text:
        print(text)
    else:
        # Show first 500 chars
        preview = text[:500] + "..." if len(text) > 500 else text
        print(preview)
    
    print("-"*80)


def view_all_chunks(store: FaissVectorStore, limit: int = 10):
    """Tampilkan semua chunks dengan limit"""
    total = len(store.metadata)
    print(f"\n📚 Total chunks in vector store: {total}")
    print(f"Showing first {min(limit, total)} chunks:\n")
    
    for i, chunk in enumerate(store.metadata[:limit]):
        print_chunk(chunk, show_full_text=False)
    
    if total > limit:
        print(f"\n... dan {total - limit} chunks lainnya")
        print(f"Gunakan --limit {total} untuk lihat semua")


def view_chunk_by_id(store: FaissVectorStore, chunk_id: int):
    """Tampilkan chunk spesifik by ID"""
    if chunk_id < 0 or chunk_id >= len(store.metadata):
        print(f"❌ Error: Chunk ID {chunk_id} tidak valid")
        print(f"   Valid range: 0 - {len(store.metadata) - 1}")
        return
    
    chunk = store.metadata[chunk_id]
    print_chunk(chunk, show_full_text=True)


def filter_by_source(store: FaissVectorStore, source_filter: str):
    """Filter chunks by source filename"""
    matching_chunks = [
        chunk for chunk in store.metadata 
        if source_filter.lower() in chunk.get('source', '').lower()
    ]
    
    if not matching_chunks:
        print(f"❌ Tidak ada chunks dari source: {source_filter}")
        print(f"\nSource yang tersedia:")
        sources = Counter([os.path.basename(c['source']) for c in store.metadata])
        for src, count in sources.items():
            print(f"  • {src} ({count} chunks)")
        return
    
    print(f"\n📚 Found {len(matching_chunks)} chunks dari source: {source_filter}\n")
    
    for chunk in matching_chunks:
        print_chunk(chunk, show_full_text=False)


def search_in_chunks(store: FaissVectorStore, keyword: str, limit: int = 10):
    """Search keyword dalam chunk text"""
    matching_chunks = []
    
    for chunk in store.metadata:
        text = chunk.get('text', '').lower()
        if keyword.lower() in text:
            # Find position of keyword
            pos = text.find(keyword.lower())
            chunk_copy = chunk.copy()
            chunk_copy['keyword_position'] = pos
            matching_chunks.append(chunk_copy)
    
    if not matching_chunks:
        print(f"❌ Keyword '{keyword}' tidak ditemukan dalam chunks")
        return
    
    print(f"\n🔍 Found {len(matching_chunks)} chunks containing '{keyword}'")
    print(f"Showing first {min(limit, len(matching_chunks))} matches:\n")
    
    for chunk in matching_chunks[:limit]:
        print("\n" + "="*80)
        print(f"📄 CHUNK #{chunk.get('chunk_id', 'N/A')} - {os.path.basename(chunk.get('source', 'N/A'))}")
        print("="*80)
        
        # Show context around keyword
        text = chunk.get('text', '')
        pos = chunk.get('keyword_position', 0)
        
        # Extract context (200 chars before and after)
        start = max(0, pos - 200)
        end = min(len(text), pos + len(keyword) + 200)
        context = text[start:end]
        
        # Highlight keyword
        context_highlighted = context.replace(
            keyword.lower(), 
            f">>>{keyword.upper()}<<<"
        )
        
        print(f"Source: {chunk.get('source', 'N/A')} - Page {chunk.get('page_label', 'N/A')}")
        print(f"\nContext:")
        print("-"*80)
        print(context_highlighted)
        print("-"*80)


def show_statistics(store: FaissVectorStore):
    """Tampilkan statistik vector store"""
    print("\n" + "="*80)
    print("📊 VECTOR STORE STATISTICS")
    print("="*80)
    
    total_chunks = len(store.metadata)
    print(f"\nTotal Chunks      : {total_chunks}")
    
    # Source statistics
    sources = [os.path.basename(c['source']) for c in store.metadata]
    source_counts = Counter(sources)
    print(f"\nTotal Documents   : {len(source_counts)}")
    print(f"\nChunks per Document:")
    for src, count in source_counts.most_common():
        print(f"  • {src:<40} : {count:>4} chunks")
    
    # Chunk size statistics
    sizes = [c.get('chunk_size', 0) for c in store.metadata]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    print(f"\nChunk Size Statistics:")
    print(f"  • Average  : {avg_size:.0f} characters")
    print(f"  • Min      : {min(sizes)} characters")
    print(f"  • Max      : {max(sizes)} characters")
    
    # Pages statistics
    pages = [c.get('page') for c in store.metadata if c.get('page')]
    if pages:
        print(f"\nPage Coverage:")
        print(f"  • Total pages with chunks : {len(set(pages))}")
        print(f"  • Page range              : {min(pages)} - {max(pages)}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="View chunks in FAISS vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/view_chunks.py                          # View first 10 chunks
  python scripts/view_chunks.py --stats                  # Show statistics
  python scripts/view_chunks.py --chunk_id 5             # View specific chunk
  python scripts/view_chunks.py --source climate.pdf    # Filter by source
  python scripts/view_chunks.py --search "RAG"           # Search keyword
  python scripts/view_chunks.py --limit 20               # View 20 chunks
        """
    )
    
    parser.add_argument("--store", default="faiss_store",
                      help="Path to vector store (default: faiss_store)")
    parser.add_argument("--chunk_id", type=int,
                      help="View specific chunk by ID")
    parser.add_argument("--source",
                      help="Filter chunks by source filename")
    parser.add_argument("--search",
                      help="Search keyword in chunks")
    parser.add_argument("--limit", type=int, default=10,
                      help="Number of chunks to display (default: 10)")
    parser.add_argument("--stats", action="store_true",
                      help="Show vector store statistics")
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*80)
    print("🔍 VECTOR STORE CHUNK VIEWER")
    print("="*80)
    
    # Load vector store
    print(f"\n[LOADING] Vector store from: {args.store}")
    try:
        store = FaissVectorStore(args.store)
        store.load()
        print(f"✅ Loaded {len(store.metadata)} chunks")
    except FileNotFoundError:
        print(f"❌ ERROR: Vector store not found at {args.store}")
        print(f"   Run: python scripts/rebuild_with_metadata.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
    
    # Execute based on arguments
    if args.stats:
        show_statistics(store)
    elif args.chunk_id is not None:
        view_chunk_by_id(store, args.chunk_id)
    elif args.source:
        filter_by_source(store, args.source)
    elif args.search:
        search_in_chunks(store, args.search, args.limit)
    else:
        view_all_chunks(store, args.limit)
    
    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
