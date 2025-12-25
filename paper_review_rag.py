# -*- coding: utf-8 -*-
"""
RAG Complete: Retrieval + Generation untuk Review Paper
Workflow:
1. Upload PDF paper baru
2. Sistem otomatis index ke vector store
3. User query tentang paper
4. Sistem retrieve relevant chunks + generate answer dengan LLM
"""
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
import os

class PaperReviewRAG:
    """RAG System untuk review paper terbaru"""
    
    def __init__(self, vector_store_path="faiss_store"):
        self.store = FaissVectorStore(vector_store_path)
        try:
            self.store.load()
            print(f"[INFO] Loaded existing vector store with {len(self.store.metadata)} chunks")
        except FileNotFoundError:
            print("[INFO] No existing vector store found. Will create new one.")
    
    def add_new_paper(self, pdf_path: str):
        """
        Upload paper baru ke vector store
        
        Args:
            pdf_path: Path ke PDF paper (e.g., "data/pdf/new_paper.pdf")
        """
        print(f"\n[STEP 1] Loading new paper: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"[ERROR] File not found: {pdf_path}")
            return False
        
        # Load hanya file ini
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        print(f"[INFO] Loaded {len(docs)} pages from paper")
        
        # Build/update vector store
        print(f"[STEP 2] Adding to vector store...")
        if self.store.index is None:
            # First time - build new
            self.store.build_from_documents(docs)
        else:
            # Add to existing store
            from src.embedding import EmbeddingPipeline
            emb_pipe = EmbeddingPipeline(
                model_name=self.store.embedding_model,
                chunk_size=self.store.chunk_size,
                chunk_overlap=self.store.chunk_overlap
            )
            
            chunks = emb_pipe.chunk_documents(docs)
            embeddings = emb_pipe.embed_chunks(chunks)
            
            # Build metadata
            metadatas = []
            for i, chunk in enumerate(chunks):
                meta = {
                    "text": chunk.page_content,
                    "chunk_id": len(self.store.metadata) + i,
                    "chunk_size": len(chunk.page_content),
                    "source": chunk.metadata.get("source", pdf_path),
                    "page": chunk.metadata.get("page"),
                    "page_label": chunk.metadata.get("page_label"),
                    "total_pages": chunk.metadata.get("total_pages"),
                    "doc_author": chunk.metadata.get("author"),
                    "doc_title": chunk.metadata.get("title"),
                    "creation_date": chunk.metadata.get("creationdate"),
                    "creator": chunk.metadata.get("creator"),
                    "original_metadata": chunk.metadata
                }
                metadatas.append(meta)
            
            self.store.add_embeddings(embeddings, metadatas)
            self.store.save()
        
        print(f"[SUCCESS] Paper added to vector store!")
        print(f"[INFO] Total chunks in store: {len(self.store.metadata)}")
        return True
    
    def ask_about_paper(self, question: str, top_k: int = 5, source_filter: str = None):
        """
        Tanya tentang paper
        
        Args:
            question: Pertanyaan tentang paper
            top_k: Berapa chunks yang di-retrieve
            source_filter: Filter by source file (optional)
        
        Returns:
            dict dengan retrieved_chunks dan answer (if LLM available)
        """
        print(f"\n{'='*70}")
        print(f"QUESTION: {question}")
        print('='*70)
        
        # STEP 1: RETRIEVAL
        print(f"\n[RETRIEVAL] Searching for relevant chunks...")
        results = self.store.query(question, top_k=top_k)
        
        # Filter by source if specified
        if source_filter:
            results = [r for r in results if source_filter in r.get('source', '')]
            print(f"[FILTER] Filtered to {len(results)} chunks from: {source_filter}")
        
        # Display retrieved chunks
        print(f"\n[RETRIEVED] Top {len(results)} relevant chunks:")
        for i, result in enumerate(results, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Source: {result['source']}")
            print(f"Page: {result.get('page_label', 'N/A')}")
            print(f"Similarity: {result['similarity_score']:.4f}")
            print(f"Text: {result['text'][:200]}...")
        
        # STEP 2: GENERATION (With LLM)
        # Note: Untuk demo, saya akan tunjukkan struktur tanpa actual LLM call
        # Anda bisa integrate dengan OpenAI, HuggingFace, atau local LLM
        
        print(f"\n{'='*70}")
        print("[GENERATION] Building answer from retrieved chunks...")
        print('='*70)
        
        # Combine retrieved chunks as context
        context = "\n\n".join([
            f"[Source: {r['source']}, Page {r.get('page_label', 'N/A')}]\n{r['text']}"
            for r in results
        ])
        
        # Build prompt for LLM
        prompt = self._build_prompt(question, context)
        
        print("\n[PROMPT FOR LLM]")
        print("-" * 70)
        print(prompt[:500] + "...")
        print("-" * 70)
        
        # TODO: Call LLM here (OpenAI, HuggingFace, local model)
        # answer = call_llm(prompt)
        
        print("\n[INFO] To get actual answer, integrate with LLM:")
        print("  Option 1: OpenAI API (GPT-3.5/GPT-4)")
        print("  Option 2: HuggingFace models")
        print("  Option 3: Local LLM (Ollama, LlamaCpp)")
        print("  See: rag_with_llm.py for implementation examples")
        
        return {
            "question": question,
            "retrieved_chunks": results,
            "context": context,
            "prompt": prompt,
            "answer": "[LLM integration needed - see rag_with_llm.py]"
        }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt untuk LLM"""
        prompt = f"""You are a helpful research assistant helping to review and analyze academic papers.

Based on the following context from research papers, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a clear and concise answer based on the context
- If the context doesn't contain enough information, say so
- Include citations (source and page number) for your claims
- If there are multiple perspectives in the papers, mention them

ANSWER:"""
        return prompt
    
    def list_papers(self):
        """List all papers in vector store"""
        if not self.store.metadata:
            print("[INFO] No papers in vector store yet")
            return
        
        from collections import Counter
        sources = [m['source'] for m in self.store.metadata]
        paper_stats = Counter(sources)
        
        print(f"\n{'='*70}")
        print(f"PAPERS IN VECTOR STORE ({len(paper_stats)} files)")
        print('='*70)
        
        for source, count in paper_stats.items():
            # Get sample metadata
            sample = next(m for m in self.store.metadata if m['source'] == source)
            print(f"\n📄 {os.path.basename(source)}")
            print(f"   Path: {source}")
            print(f"   Chunks: {count}")
            print(f"   Pages: {sample.get('total_pages', 'N/A')}")
            print(f"   Created: {sample.get('creation_date', 'N/A')}")


# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize RAG system
    rag = PaperReviewRAG("faiss_store")
    
    # List existing papers
    rag.list_papers()
    
    # Example 1: Add new paper (uncomment to use)
    # print("\n" + "="*70)
    # print("EXAMPLE: Adding New Paper")
    # print("="*70)
    # rag.add_new_paper("data/pdf/your_new_paper.pdf")
    
    # Example 2: Ask about papers
    print("\n" + "="*70)
    print("EXAMPLE: Asking About Papers")
    print("="*70)
    
    questions = [
        "What is the main contribution of the climate change paper?",
        "What methods are used in sentiment analysis?",
        "What are the key findings about topic modeling?"
    ]
    
    # Ask first question
    result = rag.ask_about_paper(questions[0], top_k=3)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Retrieved {len(result['retrieved_chunks'])} relevant chunks")
    print(f"✅ Built context with {len(result['context'])} characters")
    print(f"✅ Generated prompt for LLM")
    print(f"⚠️  Need LLM integration for final answer")
    
    # Example 3: Filter by specific paper
    print("\n" + "="*70)
    print("EXAMPLE: Query Specific Paper")
    print("="*70)
    result = rag.ask_about_paper(
        "What is the methodology?",
        top_k=3,
        source_filter="2211.03533v1.pdf"  # Specific paper
    )
