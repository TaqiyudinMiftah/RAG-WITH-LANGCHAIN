# -*- coding: utf-8 -*-
"""
RAG dengan Google Gemini API - Siap Pakai!
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from rag_with_llm import RAGWithLLM

# Load environment variables
load_dotenv()

print("="*70)
print("RAG SYSTEM - PAPER REVIEW dengan GOOGLE GEMINI")
print("="*70)

# Initialize dengan Gemini API (read from .env)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[ERROR] GEMINI_API_KEY not found in .env file!")
    print("Please create .env file with: GEMINI_API_KEY=your-key-here")
    exit(1)

rag = RAGWithLLM("faiss_store", llm_provider="gemini", api_key=api_key)

# Daftar pertanyaan untuk demo
questions = [
    "What is the main contribution of the climate change sentiment paper?",
    "What methodology was used in the sentiment analysis research?",
    "What datasets were used in these papers?",
    "What are the key findings about topic modeling in climate change discussions?",
]

print("\n[INFO] Menggunakan Google Gemini Pro (FREE)")
print("[INFO] Ready untuk review paper!\n")

# Tanya pertanyaan pertama
question = questions[0]
print("="*70)
print(f"QUESTION: {question}")
print("="*70)

result = rag.query(question, top_k=5)

print(f"\n📝 ANSWER:")
print("-"*70)
print(result['answer'])
print("-"*70)

print(f"\n📚 SOURCES:")
for i, src in enumerate(result['sources'], 1):
    print(f"  {i}. {src['source']} (Page {src['page']})")

print("\n" + "="*70)
print("Demo selesai! Sistem siap untuk pertanyaan lainnya.")
print("="*70)

# Uncomment untuk mencoba pertanyaan lain
print("\n[INFO] Untuk mencoba pertanyaan lain, edit script ini atau:")
print("""
# Interactive mode:
from rag_with_llm import RAGWithLLM

api_key = "AIzaSyBsyQhvSFn5TcJr8dEA-z4lAGIeVCdIE1M"
rag = RAGWithLLM("faiss_store", llm_provider="gemini", api_key=api_key)

# Tanya apa saja
question = "What are the limitations mentioned in the papers?"
result = rag.query(question, top_k=5)
print(result['answer'])
""")
