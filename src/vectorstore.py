import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 50):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Initialized FaissVectorStore with model {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[dict]):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} embeddings to the index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index.bin")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved FAISS index to {faiss_path} and metadata to {meta_path}.")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index.bin")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded FAISS index from {faiss_path} and metadata from {meta_path}.")
        else:
            raise FileNotFoundError("FAISS index or metadata file not found.")
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        if self.index is None:
            raise ValueError("Index not loaded. Please load or build the index first.")
        D, I = self.index.search(query_embedding.astype('float32'), top_k)
        results = []
        for distances, indices in zip(D, I):
            for dist, idx in zip(distances, indices):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(dist)
                    results.append(result)
        return results
    
    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        print(f"[INFO] Querying vector store for: {query_text}")
        query_embedding = self.model.encode([query_text]).astype('float32')
        return self.search(query_embedding, top_k=top_k)

