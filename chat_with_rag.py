# -*- coding: utf-8 -*-
"""
Interactive RAG Chat System
Tanya jawab interaktif dengan paper collection menggunakan RAG + LLM

Usage:
    python chat_with_rag.py
    
    Atau dengan parameter:
    python chat_with_rag.py --llm gemini --top_k 5
"""
from src.vectorstore import FaissVectorStore
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class InteractiveRAGChat:
    """Interactive chat interface untuk RAG system"""
    
    def __init__(self, vector_store_path="faiss_store", llm_provider="gemini", api_key=None):
        """
        Initialize RAG Chat System
        
        Args:
            vector_store_path: Path ke FAISS vector store
            llm_provider: 'gemini', 'openai', 'huggingface', atau 'ollama'
            api_key: API key (optional, bisa dari environment variable)
        """
        print("\n" + "="*70)
        print("🤖 INTERACTIVE RAG CHAT SYSTEM")
        print("="*70)
        
        # Load vector store
        print(f"\n[LOADING] Vector store dari: {vector_store_path}")
        try:
            self.store = FaissVectorStore(vector_store_path)
            self.store.load()
            print(f"✅ Loaded {len(self.store.metadata)} chunks dari vector store")
        except FileNotFoundError:
            print(f"❌ ERROR: Vector store tidak ditemukan di {vector_store_path}")
            print(f"   Jalankan: python scripts/rebuild_with_metadata.py")
            sys.exit(1)
        
        # Initialize LLM
        self.llm_provider = llm_provider
        self.api_key = api_key
        print(f"[LLM] Using {llm_provider.upper()}")
        
        # Verify API key
        if llm_provider == "gemini":
            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key or key == "your-new-gemini-api-key-here":
                print(f"❌ ERROR: GEMINI_API_KEY tidak valid")
                print(f"   1. Dapatkan API key dari: https://makersuite.google.com/app/apikey")
                print(f"   2. Tambahkan ke file .env: GEMINI_API_KEY=your-key")
                sys.exit(1)
        
        # List papers in store
        self._show_available_papers()
        
        print("\n" + "="*70)
        print("✅ RAG Chat System siap digunakan!")
        print("="*70)
    
    def _show_available_papers(self):
        """Tampilkan daftar paper yang tersedia"""
        from collections import Counter
        
        sources = [m['source'] for m in self.store.metadata]
        paper_stats = Counter(sources)
        
        print(f"\n📚 Paper yang tersedia ({len(paper_stats)} files):")
        for i, (source, count) in enumerate(paper_stats.items(), 1):
            filename = os.path.basename(source)
            print(f"   {i}. {filename} ({count} chunks)")
    
    def query(self, question: str, top_k: int = 3, show_sources: bool = True):
        """
        Query RAG system dan dapatkan jawaban
        
        Args:
            question: Pertanyaan
            top_k: Jumlah chunks yang di-retrieve
            show_sources: Tampilkan source chunks
        
        Returns:
            dict dengan answer dan sources
        """
        # STEP 1: RETRIEVAL
        if show_sources:
            print(f"\n🔍 Mencari informasi relevan...")
        
        results = self.store.query(question, top_k=top_k)
        
        if show_sources:
            print(f"✅ Ditemukan {len(results)} chunks relevan")
            print(f"\n📄 Sources:")
            for i, r in enumerate(results, 1):
                print(f"   {i}. {os.path.basename(r['source'])} - Page {r.get('page_label', 'N/A')} (score: {r['similarity_score']:.3f})")
        
        # Build context
        context = "\n\n".join([
            f"[Source: {os.path.basename(r['source'])}, Page {r.get('page_label', 'N/A')}]\n{r['text']}"
            for r in results
        ])
        
        # STEP 2: GENERATION
        if show_sources:
            print(f"\n💭 Generating answer dengan {self.llm_provider.upper()}...")
        
        answer = self._generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [{"source": os.path.basename(r['source']), 
                        "page": r.get('page_label'),
                        "score": r['similarity_score']} for r in results]
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer menggunakan LLM"""
        
        if self.llm_provider == "gemini":
            return self._generate_gemini(question, context)
        elif self.llm_provider == "openai":
            return self._generate_openai(question, context)
        elif self.llm_provider == "ollama":
            return self._generate_ollama(question, context)
        else:
            return "[ERROR] Unknown LLM provider"
    
    def _generate_gemini(self, question: str, context: str) -> str:
        """Generate dengan Google Gemini"""
        try:
            import google.generativeai as genai
            
            api_key = self.api_key or os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""You are a helpful research assistant analyzing academic papers.

Based on the following context from research papers, answer the question clearly and concisely.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a clear answer based on the context
- Include citations (source and page number)
- If the context doesn't contain enough information, say so
- Be specific and factual

ANSWER:"""
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"[ERROR] Gemini API error: {str(e)}"
    
    def _generate_openai(self, question: str, context: str) -> str:
        """Generate dengan OpenAI"""
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "[ERROR] OPENAI_API_KEY not set"
            
            client = OpenAI(api_key=api_key)
            
            prompt = f"""Based on the following context from research papers, answer the question.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear answer with citations (source and page)."""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"[ERROR] OpenAI API error: {str(e)}"
    
    def _generate_ollama(self, question: str, context: str) -> str:
        """Generate dengan Ollama (local)"""
        try:
            import ollama
            
            prompt = f"""Based on the following context from research papers, answer the question.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear and concise answer with citations."""
            
            response = ollama.generate(
                model="llama2",
                prompt=prompt,
                options={"temperature": 0.7, "num_predict": 500}
            )
            
            return response['response']
            
        except Exception as e:
            return f"[ERROR] Ollama error: {str(e)}. Is Ollama running?"
    
    def chat_loop(self, top_k: int = 3):
        """Interactive chat loop"""
        print("\n💬 Chat Mode - Ketik pertanyaan Anda (ketik 'exit' atau 'quit' untuk keluar)")
        print("="*70)
        
        while True:
            try:
                # Get user input
                print("\n" + "-"*70)
                question = input("\n❓ You: ").strip()
                
                # Check for exit commands
                if question.lower() in ['exit', 'quit', 'q', 'keluar']:
                    print("\n👋 Terima kasih telah menggunakan RAG Chat!")
                    break
                
                # Skip empty input
                if not question:
                    continue
                
                # Special commands
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if question.lower() == 'papers':
                    self._show_available_papers()
                    continue
                
                # Process query
                result = self.query(question, top_k=top_k)
                
                # Display answer
                print(f"\n🤖 Assistant:\n")
                print(result['answer'])
                
                # Show sources
                print(f"\n📚 Sources:")
                for src in result['sources']:
                    print(f"   • {src['source']} - Page {src['page']} (relevance: {src['score']:.2%})")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat dihentikan. Terima kasih!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue
    
    def _show_help(self):
        """Tampilkan help message"""
        print("\n" + "="*70)
        print("📖 HELP - Available Commands")
        print("="*70)
        print("""
Commands:
  help      - Tampilkan help ini
  papers    - List semua paper yang tersedia
  exit/quit - Keluar dari chat

Contoh Pertanyaan:
  • What is the main contribution of the climate change paper?
  • What methods are used for sentiment analysis?
  • Summarize the key findings
  • What are the limitations mentioned?
  • Compare the approaches used in different papers

Tips:
  • Tanyakan tentang methodology, findings, atau conclusions
  • Sistem akan retrieve chunks relevan dan generate answer
  • Jawaban disertai dengan citations (source dan page)
""")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive RAG Chat System")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "openai", "ollama"],
                      help="LLM provider (default: gemini)")
    parser.add_argument("--top_k", type=int, default=3,
                      help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--store", default="faiss_store",
                      help="Path to vector store (default: faiss_store)")
    
    args = parser.parse_args()
    
    try:
        # Initialize chat system
        chat = InteractiveRAGChat(
            vector_store_path=args.store,
            llm_provider=args.llm
        )
        
        # Start interactive chat
        chat.chat_loop(top_k=args.top_k)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
