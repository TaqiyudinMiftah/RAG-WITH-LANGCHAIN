# -*- coding: utf-8 -*-
"""
Interactive RAG dengan Gemini - Tanya Apa Saja tentang Paper!
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from rag_with_llm import RAGWithLLM

# Load environment variables
load_dotenv()

# Initialize
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[ERROR] GEMINI_API_KEY not found in .env file!")
    print("Please create .env file with: GEMINI_API_KEY=your-key-here")
    exit(1)

rag = RAGWithLLM("faiss_store", llm_provider="gemini", api_key=api_key)

print("="*70)
print("RAG PAPER REVIEW SYSTEM - Interactive Mode")
print("="*70)
print("\nSystem ready! Gunakan Google Gemini Pro (FREE)")
print("Vector store loaded: 295 chunks from 6 papers")
print("\n" + "="*70)

# Contoh pertanyaan
example_questions = [
    "What is the main contribution of the climate change sentiment paper?",
    "What methodology was used in the sentiment analysis research?",
    "What datasets were used in these papers?",
    "What are the key findings about topic modeling?",
    "What machine learning models were evaluated?",
    "What are the limitations mentioned in the papers?",
    "How was the data collected and preprocessed?",
]

print("\nContoh Pertanyaan:")
for i, q in enumerate(example_questions, 1):
    print(f"  {i}. {q}")

print("\n" + "="*70)
print("DEMO: Menjawab 3 Pertanyaan")
print("="*70)

# Tanya 3 pertanyaan
for i, question in enumerate(example_questions[:3], 1):
    print(f"\n{'='*70}")
    print(f"QUESTION {i}: {question}")
    print('='*70)
    
    result = rag.query(question, top_k=5)
    
    print(f"\nANSWER:")
    print("-"*70)
    print(result['answer'])
    print("-"*70)
    
    print(f"\nSources ({len(result['sources'])}):")
    for src in result['sources'][:3]:  # Show top 3 sources
        print(f"  - {src['source'].split('\\')[-1]} (Page {src['page']})")
    
    print()

print("="*70)
print("Demo Complete!")
print("="*70)
print("""
CARA PAKAI:

1. Untuk bertanya secara interactive:
   python -i test_gemini_interactive.py
   
   Lalu di Python shell:
   >>> result = rag.query("Your question here?")
   >>> print(result['answer'])

2. Dalam script:
   from rag_with_llm import RAGWithLLM
   
   rag = RAGWithLLM("faiss_store", llm_provider="gemini", api_key="your-key")
   result = rag.query("Your question")
   print(result['answer'])

3. Upload paper baru:
   from paper_review_rag import PaperReviewRAG
   
   rag_system = PaperReviewRAG("faiss_store")
   rag_system.add_new_paper("data/pdf/new_paper.pdf")
   
   # Lalu tanya tentang paper baru
   result = rag.query("What is this new paper about?")
""")
