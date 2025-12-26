# -*- coding: utf-8 -*-
"""
RAG dengan LLM Integration - 3 Options
1. OpenAI API (GPT-3.5/GPT-4)
2. HuggingFace (Free, but need API key)
3. Local LLM (Ollama - completely free and private)
"""
from src.vectorstore import FaissVectorStore
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RAGWithLLM:
    """RAG System dengan berbagai pilihan LLM"""
    
    def __init__(self, vector_store_path="faiss_store", llm_provider="gemini", api_key=None):
        self.store = FaissVectorStore(vector_store_path)
        self.store.load()
        self.llm_provider = llm_provider
        self.api_key = api_key
        print(f"[INFO] Initialized RAG with {llm_provider} LLM")
    
    def query(self, question: str, top_k: int = 3):
        """Query dengan RAG: Retrieve + Generate"""
        
        # STEP 1: RETRIEVAL
        print(f"\n[RETRIEVAL] Searching for: {question}")
        results = self.store.query(question, top_k=top_k)
        
        # Build context
        context = "\n\n".join([
            f"[Source: {r['source']}, Page {r.get('page_label', 'N/A')}]\n{r['text']}"
            for r in results
        ])
        
        # STEP 2: GENERATION
        print(f"[GENERATION] Generating answer using {self.llm_provider}...")
        
        if self.llm_provider == "gemini":
            answer = self._generate_gemini(question, context)
        elif self.llm_provider == "openai":
            answer = self._generate_openai(question, context)
        elif self.llm_provider == "huggingface":
            answer = self._generate_huggingface(question, context)
        elif self.llm_provider == "ollama":
            answer = self._generate_ollama(question, context)
        else:
            answer = "[ERROR] Unknown LLM provider"
        
        return {
            "question": question,
            "answer": answer,
            "sources": [{"source": r['source'], "page": r.get('page_label')} 
                       for r in results]
        }
    
    # ========================================================================
    # OPTION 1: Google Gemini API (FREE & POWERFUL)
    # ========================================================================
    def _generate_gemini(self, question: str, context: str) -> str:
        """
        Generate answer using Google Gemini API
        
        Setup:
        1. pip install google-generativeai
        2. Get FREE API key from: https://makersuite.google.com/app/apikey
        3. Set environment variable: GEMINI_API_KEY="your-key"
           OR pass directly: RAGWithLLM(api_key="your-key")
        
        Cost: FREE (generous quota)
        Model: Gemini 1.5 Flash (fast & capable)
        """
        try:
            import google.generativeai as genai
            
            # Get API key
            api_key = self.api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return "[ERROR] GEMINI_API_KEY not set. Get free key from: https://makersuite.google.com/app/apikey"
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')  # Stable Gemini model
            
            # Build prompt
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
            
            # Generate response
            response = model.generate_content(prompt)
            return response.text
            
        except ImportError:
            return """[ERROR] google-generativeai not installed.

Install with: pip install google-generativeai

Get FREE API key from: https://makersuite.google.com/app/apikey
"""
        except Exception as e:
            return f"[ERROR] Gemini API error: {str(e)}"
    
    # ========================================================================
    # OPTION 2: OpenAI API (GPT-3.5/GPT-4)
    # ========================================================================
    def _generate_openai(self, question: str, context: str) -> str:
        """
        Generate answer using OpenAI API
        
        Setup:
        1. pip install openai
        2. Set environment variable: OPENAI_API_KEY="your-key"
        3. Get API key from: https://platform.openai.com/api-keys
        
        Cost: ~$0.002 per query (GPT-3.5-turbo)
        """
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "[ERROR] OPENAI_API_KEY not set. Set with: export OPENAI_API_KEY='your-key'"
            
            client = OpenAI(api_key=api_key)
            
            prompt = f"""Based on the following context from research papers, answer the question.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear answer with citations (source and page)."""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" for better quality
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            return "[ERROR] openai not installed. Run: pip install openai"
        except Exception as e:
            return f"[ERROR] OpenAI API error: {str(e)}"
    
    # ========================================================================
    # OPTION 3: HuggingFace API (Free tier available)
    # ========================================================================
    def _generate_huggingface(self, question: str, context: str) -> str:
        """
        Generate answer using HuggingFace Inference API
        
        Setup:
        1. pip install huggingface_hub
        2. Get free API key from: https://huggingface.co/settings/tokens
        3. Set: export HUGGINGFACE_API_KEY='your-key'
        
        Cost: FREE (with rate limits)
        """
        try:
            from huggingface_hub import InferenceClient
            
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                return "[ERROR] HUGGINGFACE_API_KEY not set"
            
            client = InferenceClient(token=api_key)
            
            prompt = f"""Based on this context, answer the question:

Context: {context[:1000]}...

Question: {question}

Answer:"""
            
            response = client.text_generation(
                prompt,
                model="mistralai/Mistral-7B-Instruct-v0.2",
                max_new_tokens=300,
                temperature=0.7
            )
            
            return response
            
        except ImportError:
            return "[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub"
        except Exception as e:
            return f"[ERROR] HuggingFace API error: {str(e)}"
    
    # ========================================================================
    # OPTION 4: Local LLM with Ollama (FREE & PRIVATE)
    # ========================================================================
    def _generate_ollama(self, question: str, context: str) -> str:
        """
        Generate answer using local Ollama
        
        Setup:
        1. Install Ollama: https://ollama.ai
        2. Pull model: ollama pull llama2  (or mistral, codellama, etc)
        3. Start server: ollama serve
        4. pip install ollama
        
        Pros:
        - Completely FREE
        - No API key needed
        - Data stays local (privacy)
        - No rate limits
        
        Cons:
        - Need to run locally
        - Requires good CPU/GPU
        """
        try:
            import ollama
            
            prompt = f"""Based on the following context from research papers, answer the question.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear and concise answer with citations."""
            
            response = ollama.generate(
                model="llama2",  # or "mistral", "codellama", etc
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": 500
                }
            )
            
            return response['response']
            
        except ImportError:
            return """[ERROR] ollama not installed. 

Setup Ollama (Recommended - FREE & LOCAL):
1. Download from: https://ollama.ai
2. Install Ollama
3. Open terminal and run: ollama pull llama2
4. Install Python package: pip install ollama
5. Run this script again

Models available:
- llama2 (7B) - Good balance
- mistral (7B) - Fast and capable
- llama2:13b - Better quality
- codellama - For code tasks
"""
        except Exception as e:
            return f"[ERROR] Ollama error: {str(e)}. Is Ollama running? Start with: ollama serve"


# ============================================================================
# DEMO & COMPARISON
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RAG WITH LLM - SETUP GUIDE & DEMO")
    print("="*70)
    
    print("""
PILIHAN LLM:

1. Google Gemini (RECOMMENDED - FREE & POWERFUL)
   ✅ GRATIS dengan quota generous
   ✅ Kualitas excellent (Gemini 1.5 Flash)
   ✅ Mudah setup
   ✅ Fast response
   ❌ Data terkirim ke Google
   Setup: pip install google-generativeai
   Get key: https://makersuite.google.com/app/apikey

2. OpenAI (GPT-3.5/GPT-4)
   ✅ Kualitas terbaik
   ✅ Mudah setup
   ❌ Berbayar (~$0.002/query)
   ❌ Data terkirim ke OpenAI
   Setup: export OPENAI_API_KEY='your-key'

3. HuggingFace API
   ✅ FREE tier tersedia
   ✅ Banyak model
   ❌ Rate limit pada free tier
   ❌ Data terkirim ke HuggingFace
   Setup: export HUGGINGFACE_API_KEY='your-key'

4. Ollama (Local LLM)
   ✅ GRATIS 100%
   ✅ Data tetap lokal (privacy)
   ✅ Tidak ada rate limit
   ❌ Perlu install dan run lokal
   Setup: https://ollama.ai

""")
    
    print("="*70)
    print("DEMO: RAG Query with Gemini")
    print("="*70)
    
    # Demo dengan Gemini (FREE & POWERFUL)
    print("\nUsing Google Gemini API (FREE)...")
    
    # IMPORTANT: DO NOT hardcode API keys in source code!
    # Use environment variable instead for security
    
    # Option 1: Set environment variable first
    # In terminal: export GEMINI_API_KEY="your-key-here"
    # Or in code: os.environ["GEMINI_API_KEY"] = "your-key-here"
    rag = RAGWithLLM("faiss_store", llm_provider="gemini")
    
    # Option 2: Pass API key directly (only for local testing)
    # api_key = "your-api-key-here"  # Replace with your actual key
    # rag = RAGWithLLM("faiss_store", llm_provider="gemini", api_key=api_key)
    
    question = "What is the main contribution of the climate change sentiment paper?"
    result = rag.query(question, top_k=3)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for src in result['sources']:
        print(f"  - {src['source']} (page {src['page']})")
    
    print("\n" + "="*70)
    print("UNTUK MENGGUNAKAN:")
    print("="*70)
    print("""
# Pilih salah satu:

# Option 1: Google Gemini (RECOMMENDED - FREE)
api_key = "your-gemini-api-key"
rag = RAGWithLLM("faiss_store", llm_provider="gemini", api_key=api_key)
result = rag.query("Your question here")

# Option 2: OpenAI
rag = RAGWithLLM("faiss_store", llm_provider="openai")
result = rag.query("Your question here")

# Option 3: HuggingFace
rag = RAGWithLLM("faiss_store", llm_provider="huggingface")
result = rag.query("Your question here")

# Option 4: Ollama (Local)
rag = RAGWithLLM("faiss_store", llm_provider="ollama")
result = rag.query("Your question here")

print(result['answer'])
""")
