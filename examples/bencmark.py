import faiss
import numpy as np
import time

print("="*80)
print("FAISS INDEX BENCHMARK - OPTIMIZED PARAMETERS")
print("="*80)

# Generate test data
d = 384
n = 50000
queries = 100

print(f"\nDataset: {n:,} vectors, {d} dimensions, {queries} queries")
print(f"Note: Using random data (real embeddings will have better recall)\n")

vectors = np.random.randn(n, d).astype('float32')
query_vectors = np.random.randn(queries, d).astype('float32')

# Normalize vectors (makes it more like real embeddings)
faiss.normalize_L2(vectors)
faiss.normalize_L2(query_vectors)

# 1. IndexFlatL2 (Baseline - Ground Truth)
print("-"*80)
print("1. IndexFlatL2 (Exact Search - Baseline)")
print("-"*80)
index_flat = faiss.IndexFlatL2(d)
index_flat.add(vectors)

start = time.time()
D_flat, I_flat = index_flat.search(query_vectors, 10)
time_flat = (time.time() - start) / queries * 1000
print(f"   Time: {time_flat:.3f} ms/query")
print(f"   Recall: 100% (ground truth)")
print(f"   Speedup: 1.0×")

# 2. IndexIVFFlat - OPTIMIZED PARAMETERS
print("\n" + "-"*80)
print("2. IndexIVFFlat (Approximate Search)")
print("-"*80)

# Better parameters for random data
nlist = 50  # Fewer clusters = bigger clusters = more likely to find neighbors
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

print(f"   Training with K-means (nlist={nlist})...")
index_ivf.train(vectors)
index_ivf.add(vectors)

# Test different nprobe values
for nprobe in [5, 10, 20]:
    index_ivf.nprobe = nprobe
    
    start = time.time()
    D_ivf, I_ivf = index_ivf.search(query_vectors, 10)
    time_ivf = (time.time() - start) / queries * 1000
    
    # Calculate recall@10
    recall_ivf = np.mean([len(set(I_flat[i]) & set(I_ivf[i])) / 10 
                          for i in range(queries)])
    
    speedup = time_flat / time_ivf
    search_pct = (nprobe / nlist) * 100
    
    print(f"   nprobe={nprobe:2d} (search {search_pct:4.1f}% of clusters):")
    print(f"     - Time: {time_ivf:.3f} ms/query")
    print(f"     - Recall: {recall_ivf*100:.1f}%")
    print(f"     - Speedup: {speedup:.1f}×")

# 3. IndexHNSWFlat - OPTIMIZED PARAMETERS
print("\n" + "-"*80)
print("3. IndexHNSWFlat (Graph-based Search)")
print("-"*80)

# Better parameters
M = 32
index_hnsw = faiss.IndexHNSWFlat(d, M)

print(f"   Building HNSW graph (M={M})...")
index_hnsw.add(vectors)

# Test different efSearch values
for efSearch in [32, 64, 128, 256]:
    index_hnsw.hnsw.efSearch = efSearch
    
    start = time.time()
    D_hnsw, I_hnsw = index_hnsw.search(query_vectors, 10)
    time_hnsw = (time.time() - start) / queries * 1000
    
    recall_hnsw = np.mean([len(set(I_flat[i]) & set(I_hnsw[i])) / 10 
                           for i in range(queries)])
    
    speedup = time_flat / time_hnsw
    
    print(f"   efSearch={efSearch:3d}:")
    print(f"     - Time: {time_hnsw:.3f} ms/query")
    print(f"     - Recall: {recall_hnsw*100:.1f}%")
    print(f"     - Speedup: {speedup:.1f}×")

# Summary
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)
print("""
For your RAG project (real embeddings, not random data):

1. Current (< 50K chunks):
   → IndexFlatL2: Perfect! Fast enough and 100% accurate.

2. Scale to 100K-500K chunks:
   → IndexHNSWFlat (M=32, efSearch=64-128)
   → Expected: 10-30× speedup, 98-99% recall
   
3. Scale to 1M+ chunks:
   → IndexIVFFlat (nlist=4096, nprobe=64-128)
   → Or migrate to Qdrant/Milvus for distributed search

Note: Real embeddings have better clustering structure than random data,
      so expect 5-10% higher recall in production!
""")