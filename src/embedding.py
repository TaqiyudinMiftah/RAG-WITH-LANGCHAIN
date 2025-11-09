from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        return chunks
    
    def embed_chunks(self, documents: List[Any]) -> np.ndarray:
        texts = [doc.page_content for doc in documents]
        print(f"Embedding {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings