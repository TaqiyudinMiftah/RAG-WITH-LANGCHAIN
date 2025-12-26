#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
Quick Start Launcher untuk RAG System
Script untuk memudahkan akses ke berbagai fungsi RAG
"""

echo "========================================================================"
echo "🚀 RAG WITH LANGCHAIN - QUICK LAUNCHER"
echo "========================================================================"
echo ""
echo "Pilih opsi yang ingin Anda jalankan:"
echo ""
echo "1. 💬 Interactive Chat         - Tanya jawab dengan paper collection"
echo "2. 📚 Build Vector Store        - Index paper PDF ke database"
echo "3. 🔍 Inspect Vector Store      - Lihat isi vector store"
echo "4. 📊 Paper Review Demo         - Demo review paper"
echo "5. 🧪 Test RAG with LLM         - Test RAG dengan berbagai LLM"
echo "6. ❌ Exit"
echo ""
echo "========================================================================"

read -p "Masukkan pilihan (1-6): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching Interactive Chat..."
        echo "========================================================================"
        uv run python chat_with_rag.py
        ;;
    2)
        echo ""
        echo "🚀 Building Vector Store..."
        echo "========================================================================"
        echo "ℹ️  Pastikan PDF paper ada di folder: data/pdf/"
        echo ""
        uv run python scripts/rebuild_with_metadata.py
        ;;
    3)
        echo ""
        echo "🚀 Inspecting Vector Store..."
        echo "========================================================================"
        uv run python scripts/inspect_metadata.py
        ;;
    4)
        echo ""
        echo "🚀 Running Paper Review Demo..."
        echo "========================================================================"
        uv run python examples/paper_review_rag.py
        ;;
    5)
        echo ""
        echo "🚀 Testing RAG with LLM..."
        echo "========================================================================"
        uv run python examples/rag_with_llm.py
        ;;
    6)
        echo ""
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Invalid option: $choice"
        echo "Please choose 1-6"
        exit 1
        ;;
esac
