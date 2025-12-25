from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

if __name__ == "__main__":
    # Load and query existing vector store
    store = FaissVectorStore("faiss_store")
    store.load()
    
    # Test query
    query = "What is LangChain?"
    print(f"\nQuery: {query}")
    print("=" * 70)
    
    results = store.query(query, top_k=3)
    
    # Display results with enriched metadata
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Source: {result.get('source', 'N/A')}")
        print(f"  Page: {result.get('page_label', 'N/A')}/{result.get('total_pages', 'N/A')}")
        print(f"  Chunk ID: {result.get('chunk_id', 'N/A')}")
        print(f"  Created: {result.get('creation_date', 'N/A')}")
        print(f"  Similarity Score: {result.get('similarity_score', 'N/A'):.4f}")
        print(f"  Text: {result.get('text', '')[:150]}...")
        print("-" * 70)
